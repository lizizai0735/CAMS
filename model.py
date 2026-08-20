import argparse
import json
import os
import cv2
import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_factory import MultiTaskModelFactory
from utils import decode_heatmaps_to_normalized_coords

# ==========================================
# Task configuration: Define the number of keypoints (num_classes) for each task
# ==========================================
TASK_NUM_CLASSES = {
    "A4C": 16,
    "AOP": 4,
    "FA": 4,
    "fetal_femur": 2,
    "FUGC": 2,
    "HC": 4,
    "IVC": 2,
    "PLAX": 22,
    "PSAX": 4
}
# ==========================================


class FolderInferenceDataset(Dataset):
    """Dataset class for loading inference images organized by task folders."""

    def __init__(self, data_root: str, transforms: A.Compose = None):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.samples = []

        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")

        # Iterate through subdirectories (directory name corresponds to task_id)
        for task_id in os.listdir(data_root):
            task_dir = os.path.join(data_root, task_id)
            if not os.path.isdir(task_dir):
                continue

            for img_name in os.listdir(task_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    img_path = os.path.join(task_dir, img_name)
                    self.samples.append({
                        "image_path": img_path,
                        "task_id": task_id
                    })

        if not self.samples:
            raise ValueError(f"No images found in {data_root} or its subdirectories.")

        print(f"Dataset loaded. Total images found: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        task_id = sample["task_id"]

        image = cv2.imread(image_path)
        if image is None:
            # Fallback: skip corrupted images
            return self.__getitem__((idx + 1) % len(self))

        # Apply CLAHE preprocessing (must match the training dataset pipeline)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        image_clahe = cv2.merge((cl, a, b))

        # Convert back to RGB for Albumentations and model input
        image = cv2.cvtColor(image_clahe, cv2.COLOR_LAB2RGB)

        original_height, original_width = image.shape[:2]

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return {
            "image": image,
            "task_id": task_id,
            "task_name": "Regression",
            "image_path": image_path,
            "original_size": (original_height, original_width),
            "index": idx,
        }


def inference_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], 0)
    return {
        "image": images,
        "task_id": [item["task_id"] for item in batch],
        "task_name": [item["task_name"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
        "index": [item["index"] for item in batch],
    }


class Model:
    """Inference model for keypoint localization."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = None
        self.task_configs = None
        self.heatmap_size = (64, 64)
        self.input_size = 518

        self.transforms = A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _build_task_configs(self, unique_task_ids: list):
        task_configs = []
        for task_id in unique_task_ids:
            if task_id not in TASK_NUM_CLASSES:
                raise ValueError(f"Missing configuration: Please add '{task_id}' to TASK_NUM_CLASSES.")

            task_configs.append({
                "task_id": task_id,
                "task_name": "Regression",
                "num_classes": TASK_NUM_CLASSES[task_id],
            })
        return task_configs

    def _load_model(self, model_path: str):
        self.model = MultiTaskModelFactory(
            encoder_name="vit_small_patch14_dinov2.lvd142m",
            encoder_weights=None,
            task_configs=self.task_configs,
            heatmap_size=self.heatmap_size,
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        # Load checkpoint securely
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model_dict = self.model.state_dict()

        # Handle potential channel mismatch by dynamically cropping weights
        for k in list(checkpoint.keys()):
            if k in model_dict:
                ckpt_shape = checkpoint[k].shape
                model_shape = model_dict[k].shape
                if ckpt_shape != model_shape:
                    if len(ckpt_shape) == 4 and len(model_shape) == 4:
                        if ckpt_shape[0] == model_shape[0] and ckpt_shape[2:] == model_shape[2:]:
                            if ckpt_shape[1] > model_shape[1]:
                                print(f"Dimension mismatch fixed: Cropping {k} from {ckpt_shape[1]} to {model_shape[1]} channels.")
                                checkpoint[k] = checkpoint[k][:, :model_shape[1], :, :]

        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

    def predict(self, data_root: str, output_dir: str, model_path: str, batch_size: int = 8):
        print("=" * 60)
        print("Starting keypoint prediction...")
        print(f"Validation data path: {data_root}")
        print(f"Model weights path: {model_path}")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)
        dataset = FolderInferenceDataset(data_root=data_root, transforms=self.transforms)

        unique_tasks = list(set([sample["task_id"] for sample in dataset.samples]))
        self.task_configs = self._build_task_configs(unique_tasks)
        self._load_model(model_path)

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=inference_collate_fn
        )

        regression_results = []
        task_counts = {}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Prediction progress"):
                images = batch["image"].to(self.device)
                task_ids = batch["task_id"]
                image_paths = batch["image_path"]
                original_sizes = batch["original_size"]

                batch_unique_tasks = list(set(task_ids))
                for task_id in batch_unique_tasks:
                    task_indices = [i for i, tid in enumerate(task_ids) if tid == task_id]
                    task_images = images[task_indices]

                    # Use DSNT to obtain continuous sub-pixel coordinates
                    pred_coords_dsnt, _ = self.model(task_images, task_id=task_id, return_coords=True)

                    # Map DSNT outputs from [-1, 1] to [0, 1]
                    pred_coords_01 = (pred_coords_dsnt + 1.0) / 2.0
                    outputs = pred_coords_01.view(pred_coords_01.shape[0], -1)  # Flatten to [B, K*2]

                    for i, batch_idx in enumerate(task_indices):
                        pred = outputs[i]
                        image_path = image_paths[batch_idx]
                        original_size = original_sizes[batch_idx]
                        task_counts[task_id] = task_counts.get(task_id, 0) + 1

                        regression_results.append(
                            self._process_regression(pred, task_id, image_path, original_size)
                        )

        # Save predictions to JSON
        json_path = os.path.join(output_dir, "regression_predictions.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(regression_results, f, indent=2, ensure_ascii=False)

        print(f"\nPrediction completed. Results saved to: {json_path}")
        print("Prediction statistics per task:")
        for task_id in sorted(task_counts.keys()):
            print(f"  - {task_id}: {task_counts[task_id]} images")

    def _process_regression(self, pred, task_id, image_path, original_size):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        coords = pred.flatten().tolist()
        h, w = original_size
        pixel_coords = []
        for i in range(0, len(coords), 2):
            x_norm, y_norm = coords[i], coords[i + 1]
            pixel_coords.extend([x_norm * w, y_norm * h])

        # Generate relative path (e.g., "A4C/0001.png")
        base_filename = os.path.basename(image_path)
        relative_path = f"{task_id}/{base_filename}"

        return {
            "image_path": relative_path,
            "task_id": task_id,
            "predicted_points_normalized": coords,
            "predicted_points_pixels": pixel_coords
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for folder-based validation datasets")
    parser.add_argument("--data-root", type=str, default="./val_data",
                        help="Path to the validation data directory")
    parser.add_argument("--model-path", type=str, default="./checkpoints/best_model.pth",
                        help="Path to the model weights file")
    parser.add_argument("--output-dir", type=str, default="./predictions", help="Output directory for prediction results")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    args = parser.parse_args()

    model = Model()
    model.predict(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )