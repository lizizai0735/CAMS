import glob
import json
import os
import random
from typing import Iterator, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

# Set of task IDs designated for keypoint localization (regression) tasks
EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX", "fetal_femur", "FUGC"}


class KeypointDataset(Dataset):
    """Dataset class for labeled ultrasound keypoint localization tasks.

    Handles image path resolution, CLAHE contrast enhancement, keypoint
    coordinate normalization, and target Gaussian heatmap generation.

    Args:
        data_root (str): Root directory containing images and CSV annotation files.
        transforms (Optional[A.Compose]): Albumentations augmentation pipeline.
        heatmap_size (Tuple[int, int]): (Height, Width) dimensions for generated target heatmaps. Defaults to (64, 64).
        sigma (float): Standard deviation (Gaussian kernel width) for keypoint heatmaps. Defaults to 1.8.
    """

    def __init__(
            self,
            data_root: str,
            transforms: Optional[A.Compose] = None,
            heatmap_size: Tuple[int, int] = (64, 64),
            sigma: float = 1.8
    ):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        # ------------------------------------------------------------------
        # Step 1: Scan and count total image files present on disk
        # ------------------------------------------------------------------
        image_dir = os.path.join(self.data_root, "images")
        search_base = image_dir if os.path.exists(image_dir) else self.data_root

        print(f"Scanning disk for images in: {search_base}...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        disk_image_count = 0
        for root, _, files in os.walk(search_base):
            for f in files:
                if os.path.splitext(f)[1].lower() in image_extensions:
                    disk_image_count += 1

        print(f"--- Disk Statistics ---")
        print(f"Found {disk_image_count} images on disk in {search_base}")
        print(f"-----------------------")

        # ------------------------------------------------------------------
        # Step 2: Load and aggregate CSV annotation records
        # ------------------------------------------------------------------
        self.csv_path = os.path.join(self.data_root, "csv")
        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")

        all_csv_files = glob.glob(os.path.join(self.csv_path, "*.csv"))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")

        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)

        is_regression = dataframe["task_name"].astype(str).eq("Regression")
        is_extra_task = dataframe["task_id"].astype(str).isin(EXTRA_REGRESSION_TASK_IDS)
        self.dataframe = dataframe[is_regression | is_extra_task].reset_index(drop=True)

        if self.dataframe.empty:
            raise ValueError("No keypoint records found in CSV.")

        # ------------------------------------------------------------------
        # Step 3: Match CSV annotation records against actual disk paths
        # ------------------------------------------------------------------
        print("Pre-scanning dataset for valid image paths (CSV vs Disk)...")
        valid_count = 0

        # Store resolved absolute paths for filtering labeled images in UnlabeledDataset
        self.resolved_paths = set()

        for _, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe), desc="Checking images"):
            resolved_path = self._resolve_image_path(str(row["image_path"]), str(row["task_id"]))
            if resolved_path is not None:
                valid_count += 1
                self.resolved_paths.add(os.path.abspath(resolved_path))

        print(f"--- Labeled Dataset Statistics ---")
        print(f"Total keypoint records in CSV: {len(self.dataframe)}")
        print(f"Total records successfully matched to disk: {valid_count}")
        print(f"----------------------------------")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _resolve_image_path(self, rel_path: str, task_id: str = "") -> Optional[str]:
        """Resolves a relative image path to an absolute path on disk.

        Args:
            rel_path (str): Relative or absolute image path from CSV.
            task_id (str): Task identifier used for subfolder searching.

        Returns:
            Optional[str]: Verified absolute file path if found, else None.
        """
        if os.path.isabs(rel_path) and os.path.isfile(rel_path):
            return rel_path

        filename = os.path.basename(rel_path)
        search_root = os.path.join(self.data_root, "images")
        if not os.path.exists(search_root):
            search_root = self.data_root

        parts = os.path.normpath(rel_path).split(os.sep)
        clean_parts = [p for p in parts if p not in [".", ".."]]

        direct_attempt = os.path.join(search_root, *clean_parts)
        if os.path.isfile(direct_attempt):
            return direct_attempt

        limit_root = search_root
        if task_id:
            task_specific_root = os.path.join(search_root, task_id)
            if os.path.exists(task_specific_root):
                limit_root = task_specific_root

        for root, _, files in os.walk(limit_root):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def _generate_heatmaps(self, norm_coords: np.ndarray, num_points: int, valid_mask: List[bool]) -> np.ndarray:
        """Generates 2D Gaussian target heatmaps for given normalized keypoints.

        Args:
            norm_coords (np.ndarray): Flattened normalized coordinates [x0, y0, x1, y1, ...].
            num_points (int): Maximum number of keypoints for the task.
            valid_mask (List[bool]): Mask indicating whether each keypoint is valid/present.

        Returns:
            np.ndarray: Generated target heatmaps array of shape (num_points, height, width).
        """
        heatmap_h, heatmap_w = self.heatmap_size
        yy, xx = np.meshgrid(np.arange(heatmap_h), np.arange(heatmap_w), indexing="ij")
        heatmaps = np.zeros((num_points, heatmap_h, heatmap_w), dtype=np.float32)

        for i in range(num_points):
            if not valid_mask[i]:
                continue

            x_norm = min(max(float(norm_coords[2 * i]), 0.0), 1.0)
            y_norm = min(max(float(norm_coords[2 * i + 1]), 0.0), 1.0)
            x = x_norm * (heatmap_w - 1)
            y = y_norm * (heatmap_h - 1)
            dist2 = (xx - x) ** 2 + (yy - y) ** 2
            heatmaps[i] = np.exp(-dist2 / (2.0 * self.sigma * self.sigma)).astype(np.float32)
        return heatmaps

    def __getitem__(self, idx: int) -> dict:
        attempt_count = 0
        max_attempts = len(self.dataframe)
        current_idx = idx

        while attempt_count < max_attempts:
            record = self.dataframe.iloc[current_idx]
            task_id = str(record["task_id"])
            image_abs_path = self._resolve_image_path(str(record["image_path"]), task_id)

            if image_abs_path is not None:
                image = cv2.imread(image_abs_path)
                if image is not None:
                    # Apply CLAHE preprocessing in LAB color space
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    image_clahe = cv2.merge((cl, a, b))
                    image = cv2.cvtColor(image_clahe, cv2.COLOR_LAB2RGB)

                    num_points = int(record["num_classes"])
                    coords = []
                    valid_mask = []

                    # Extract coordinates and build validity mask
                    for i in range(1, num_points + 1):
                        col = f"point_{i}_xy"
                        if col in record and pd.notna(record[col]):
                            val = record[col]
                            pts = json.loads(val) if isinstance(val, str) else val
                            if isinstance(pts, list) and len(pts) >= 2:
                                coords.extend(pts[:2])
                                valid_mask.append(True)
                            else:
                                coords.extend([0.0, 0.0])
                                valid_mask.append(False)
                        else:
                            coords.extend([0.0, 0.0])
                            valid_mask.append(False)

                    kpts = [(float(coords[j]), float(coords[j + 1])) for j in range(0, 2 * num_points, 2)]

                    # Apply data augmentations if defined
                    if self.transforms:
                        transformed = self.transforms(image=image, keypoints=kpts)
                        image = transformed["image"]
                        kpts = transformed["keypoints"]

                    if isinstance(image, torch.Tensor):
                        out_h, out_w = image.shape[1], image.shape[2]
                    else:
                        out_h, out_w = image.shape[:2]

                    # Normalize target keypoint coordinates to [0, 1]
                    label = np.zeros(2 * num_points, dtype=np.float32)
                    for i, (x, y) in enumerate(kpts):
                        if i < num_points:
                            if not valid_mask[i]:
                                label[2 * i] = 0.0
                                label[2 * i + 1] = 0.0
                            else:
                                label[2 * i] = x / max(float(out_w - 1), 1.0)
                                label[2 * i + 1] = y / max(float(out_h - 1), 1.0)

                    label = np.clip(label, 0.0, 1.0)

                    return {
                        "image": image,
                        "label": torch.from_numpy(label).float(),
                        "heatmap": torch.from_numpy(self._generate_heatmaps(label, num_points, valid_mask)).float(),
                        "task_id": task_id,
                    }

            current_idx = (current_idx + 1) % len(self.dataframe)
            attempt_count += 1
        raise RuntimeError("No valid images found.")


class KeypointUniformSampler(Sampler[List[int]]):
    """Uniform task-level batch sampler.

    Ensures balanced task sampling by selecting a random task for each batch
    and sampling indices strictly within that task group.

    Args:
        dataset (KeypointDataset): Source keypoint dataset containing task annotations.
        batch_size (int): Size of each sampled batch.
        steps_per_epoch (Optional[int]): Total iterations per epoch. Defaults to len(dataset) // batch_size.
    """

    def __init__(self, dataset: KeypointDataset, batch_size: int, steps_per_epoch: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_by_task = {}

        print("\n--- Initializing Keypoint Sampler ---")
        for idx, task_id in enumerate(tqdm(dataset.dataframe["task_id"], desc="Grouping indices")):
            if task_id not in self.indices_by_task:
                self.indices_by_task[task_id] = []
            self.indices_by_task[task_id].append(idx)

        self.task_ids = list(self.indices_by_task.keys())
        self.steps_per_epoch = steps_per_epoch or (len(self.dataset) // self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        task_cursors = {tid: 0 for tid in self.task_ids}
        for tid in self.task_ids:
            random.shuffle(self.indices_by_task[tid])

        for _ in range(self.steps_per_epoch):
            tid = random.choice(self.task_ids)
            indices = self.indices_by_task[tid]
            if task_cursors[tid] + self.batch_size > len(indices):
                random.shuffle(indices)
                task_cursors[tid] = 0

            batch = indices[task_cursors[tid]:task_cursors[tid] + self.batch_size]
            task_cursors[tid] += self.batch_size
            yield batch

    def __len__(self) -> int:
        return self.steps_per_epoch


class UnlabeledDataset(Dataset):
    """Dataset class for unlabeled ultrasound images in semi-supervised training (e.g., Mean Teacher).

    Automatically scans data directories and strictly excludes all labeled images based on absolute paths.

    Args:
        data_root (str): Root directory containing ultrasound images.
        excluded_paths (set): Set of absolute file paths to exclude (e.g., labeled instances).
        transforms (Optional[A.Compose]): Albumentations augmentation pipeline.
    """

    def __init__(self, data_root: str, excluded_paths: set, transforms: Optional[A.Compose] = None):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.samples = []

        search_root = os.path.join(self.data_root, "images")
        if not os.path.exists(search_root):
            search_root = self.data_root

        print(f"\nScanning for Unlabeled images in: {search_root}...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        # Traverse task-specific directories
        for task_id in os.listdir(search_root):
            if task_id not in EXTRA_REGRESSION_TASK_IDS:
                continue

            task_base_dir = os.path.join(search_root, task_id)
            if not os.path.isdir(task_base_dir):
                continue

            # Recursively collect all image paths inside the task subdirectories
            for root, _, files in os.walk(task_base_dir):
                for img_name in files:
                    if os.path.splitext(img_name)[1].lower() in image_extensions:
                        abs_path = os.path.abspath(os.path.join(root, img_name))

                        # Exclude any images that are already present in the labeled dataset
                        if abs_path not in excluded_paths:
                            self.samples.append({
                                "image_path": abs_path,
                                "task_id": task_id
                            })

        print(f"--- Unlabeled Dataset Statistics ---")
        print(f"Successfully loaded {len(self.samples)} completely unlabeled images.")
        print(f"------------------------------------")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        task_id = sample["task_id"]

        image = cv2.imread(image_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        # Perform identical CLAHE preprocessing as used in the labeled dataset
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        image_clahe = cv2.merge((cl, a, b))
        image = cv2.cvtColor(image_clahe, cv2.COLOR_LAB2RGB)

        # Apply image transformations (stripping keypoint parameters for unlabeled data)
        if self.transforms:
            aug_fn = A.Compose(self.transforms.transforms)
            image = aug_fn(image=image)["image"]

        return {
            "image": image,
            "task_id": task_id,
        }


def unlabeled_collate_fn(batch: List[dict]) -> dict:
    """Collate function for DataLoader handling unlabeled image batches.

    Args:
        batch (List[dict]): List of samples returned by UnlabeledDataset.__getitem__.

    Returns:
        dict: Collated dictionary containing stacked image Tensors and task IDs.
    """
    images = torch.stack([item["image"] for item in batch], 0)
    task_ids = [item["task_id"] for item in batch]
    return {"image": images, "task_id": task_ids}