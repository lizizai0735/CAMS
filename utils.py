import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def set_seed(seed: int) -> None:
    """Sets the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def keypoint_collate_fn(batch):
    """Custom collate function for keypoint dataset batches."""
    images = torch.stack([item["image"] for item in batch], 0)
    labels = [item["label"] for item in batch]
    heatmaps = [item["heatmap"] for item in batch]
    task_ids = [item["task_id"] for item in batch]
    return {"image": images, "label": labels, "heatmap": heatmaps, "task_id": task_ids}


def decode_heatmaps_to_normalized_coords(heatmaps: torch.Tensor) -> torch.Tensor:
    """Extracts normalized coordinates [0, 1] from heatmaps using argmax."""
    bsz, num_points, h, w = heatmaps.shape
    flat_idx = heatmaps.view(bsz, num_points, -1).argmax(dim=-1)
    ys = torch.div(flat_idx, w, rounding_mode="floor").float()
    xs = (flat_idx % w).float()
    x_norm = xs / max(float(w - 1), 1.0)
    y_norm = ys / max(float(h - 1), 1.0)
    coords = torch.stack([x_norm, y_norm], dim=-1).reshape(bsz, num_points * 2)
    return coords


def calculate_mre(y_true: torch.Tensor, y_pred: torch.Tensor, valid_mask: torch.Tensor, image_size=(256, 256)) -> float:
    """
    Calculates the Mean Radial Error (MRE) in pixels.
    Integrates a valid_mask mechanism to prevent calculation errors caused by missing keypoints.
    Applies strict (w-1) and (h-1) scaling alignment to eliminate systematic coordinate shifts.
    """
    h, w = image_size
    y_true_px = y_true.detach().cpu().numpy().copy()
    y_pred_px = y_pred.detach().cpu().numpy().copy()
    mask = valid_mask.detach().cpu().numpy().copy()

    y_true_px[:, 0::2] *= (w - 1)
    y_true_px[:, 1::2] *= (h - 1)
    y_pred_px[:, 0::2] *= (w - 1)
    y_pred_px[:, 1::2] *= (h - 1)

    y_true_pts = y_true_px.reshape(y_true_px.shape[0], -1, 2)
    y_pred_pts = y_pred_px.reshape(y_pred_px.shape[0], -1, 2)

    distances = np.sqrt(np.sum((y_pred_pts - y_true_pts) ** 2, axis=-1))

    valid_distances = distances[mask > 0]
    if len(valid_distances) == 0:
        return 0.0

    return float(np.mean(valid_distances))


def evaluate_keypoint(model, val_loader, device, task_id_to_name):
    """Evaluates the model on the validation set and returns a DataFrame of metrics per task."""
    model.eval()
    task_metrics = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Validation]")
        for batch in loop:
            images = batch["image"].to(device)
            labels = batch["label"]
            task_ids = batch["task_id"]

            unique_tasks = set(task_ids)
            for task_id in unique_tasks:
                task_indices = [i for i, t in enumerate(task_ids) if t == task_id]
                task_images = images[task_indices]
                task_labels = torch.stack([labels[i] for i in task_indices], 0).to(device)

                B = task_labels.shape[0]
                K = task_labels.shape[1] // 2
                valid_mask = (task_labels.view(B, K, 2).sum(dim=-1) > 0).float()

                image_size = (int(task_images.shape[-2]), int(task_images.shape[-1]))

                pred_coords, _ = model(task_images, task_id=task_id, return_coords=True)

                pred_coords_01 = (pred_coords + 1.0) / 2.0
                bsz = pred_coords_01.shape[0]
                pred_coords_01 = pred_coords_01.view(bsz, -1)

                task_metrics[task_id]["MRE (pixels)"].append(
                    calculate_mre(task_labels, pred_coords_01, valid_mask, image_size=image_size)
                )

    results = []
    for task_id in sorted(task_metrics.keys()):
        row = {"Task ID": task_id, "Task Name": task_id_to_name.get(task_id, "Regression")}
        for metric_name, values in task_metrics[task_id].items():
            row[metric_name] = float(np.mean(values)) if values else 0.0
        results.append(row)

    return pd.DataFrame(results)