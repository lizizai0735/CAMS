import os

# Force deterministic CUDA algorithms
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
from collections import defaultdict
import importlib
import copy
import random
import cv2

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataset import KeypointDataset, KeypointUniformSampler, UnlabeledDataset, unlabeled_collate_fn
from model_factory import MultiTaskModelFactory
from utils import evaluate_keypoint, keypoint_collate_fn

# ================= Core Hyperparameters =================
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
UNLABELED_BATCH_SIZE = 4  # Batch size for unlabeled data in semi-supervised learning
NUM_EPOCHS = 50
DATA_ROOT_PATH = "./data/FU_Biometry"
ENCODER = "vit_small_patch14_dinov2.lvd142m"
ENCODER_WEIGHTS = "pretrained"
RANDOM_SEED = 42

CHECKPOINT_DIR = "./checkpoints_pro1"
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LATEST_STATE_PATH = os.path.join(CHECKPOINT_DIR, "latest_training_state.pth")

VAL_SPLIT = 0.2
HEATMAP_SIZE = (64, 64)

# Tuning: Increased Sigma (3.0) provides broader gradient guidance for specific tasks (e.g., HC/FA)
# and prevents vanishing gradients for outlier keypoints.
HEATMAP_SIGMA = 3.0
INPUT_SIZE = 518
EMA_DECAY = 0.999
UNSUP_MAX_WEIGHT = 2.0  # Maximum weight for unsupervised consistency loss

EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX", "FUGC", "fetal_femur"}

# Task-specific loss multipliers to penalize tasks with historically high MRE
TASK_WEIGHTS = {
    "HC": 3.0,
    "PSAX": 3.0,
    "A4C": 3.0,
    "FA": 2.0,
    "fetal_femur": 1.5,
    "IVC": 1.5
}


# ================= Split-Screen Image Detection =================
def is_split_screen(image_path):
    """Detects if an image is split-screen by checking for a continuous dark band in the center."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        h, w = img.shape
        col_means = np.mean(img, axis=0)
        # Check the middle 10% width; if the minimum average brightness is close to black, it's likely a split screen.
        mid_start, mid_end = int(w * 0.45), int(w * 0.55)
        if np.min(col_means[mid_start:mid_end]) < 5:
            return True
        return False
    except Exception:
        return False


# ================= Deterministic Settings =================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ================= Consistency Weight Warmup =================
def get_consistency_weight(epoch, total_epochs, max_weight=2.0):
    rampup_epochs = total_epochs * 0.4
    if epoch < rampup_epochs:
        p = max(0.0, float(epoch) / float(rampup_epochs))
        p = 1.0 - p
        return max_weight * np.exp(-5.0 * p * p)
    return max_weight


class ModelEMA:
    def __init__(self, model, decay):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            esd = self.ema.module.state_dict() if self.ema_has_module else self.ema.state_dict()
            for k, ema_v in esd.items():
                if ema_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * self.decay + (1. - self.decay) * msd[k].detach())


class TrainingLogger:
    def __init__(self, log_path: str = "training_pro_log.txt"):
        self.log_path = log_path
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=== Training Process Log ===\n")

    def log_epoch(self, epoch, train_losses_dict, val_df, avg_val_score, metric_label, is_best):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- Epoch {epoch} Average Train Loss ---\n")
            for task_id in sorted(train_losses_dict.keys()):
                avg_loss = float(np.mean(train_losses_dict[task_id]))
                f.write(f"  - {task_id}: {avg_loss:.4f}\n")
            f.write(f"\n--- Epoch {epoch} Validation Report ---\n")
            if not val_df.empty:
                f.write(val_df.to_string(index=False))
                f.write("\n")
            f.write(f"--- Average Val {metric_label} (Lower is better): {avg_val_score:.4f} ---\n")
            if is_best:
                f.write(f"-> New best model saved! {metric_label} improved to: {avg_val_score:.4f}\n")
            f.write("-" * 50 + "\n")


def _build_task_configs(dataframe):
    configs = []
    seen = set()
    for _, row in dataframe.iterrows():
        task_name = str(row["task_name"])
        task_id = str(row["task_id"])
        if task_name != "Regression" and task_id not in EXTRA_REGRESSION_TASK_IDS:
            continue
        if task_id in seen:
            continue
        seen.add(task_id)
        configs.append({
            "task_id": task_id,
            "task_name": "Regression",
            "num_classes": int(row["num_classes"]),
        })
    return configs


def _stratified_split_indices(dataframe, val_split: float, seed: int):
    rng = np.random.RandomState(seed)
    train_indices, val_indices = [], []
    for _, group in dataframe.groupby("task_id", sort=True):
        indices = np.array(group.index.to_numpy(), copy=True)
        rng.shuffle(indices)
        total = len(indices)
        val_count = int(round(total * float(val_split)))
        if total >= 2:
            val_count = max(1, min(total - 1, val_count))
        else:
            val_count = 0
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def main(args, val_split: float = VAL_SPLIT):
    metric_column = "MRE (pixels)"
    metric_label = "MRE"

    seed_everything(RANDOM_SEED)
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logger = TrainingLogger("training_pro_log.txt")

    # Tuning: Optimize data augmentation pipeline.
    # Removed ElasticTransform to preserve geometric structures (e.g., ellipses for HC/FA).
    # Reduced the intensity of ShiftScaleRotate and CoarseDropout to retain anatomical features.
    train_transforms = A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.1),
        A.CoarseDropout(max_holes=2, max_height=int(INPUT_SIZE * 0.05), max_width=int(INPUT_SIZE * 0.05), p=0.05),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    val_transforms = A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    temp_dataset = KeypointDataset(DATA_ROOT_PATH, transforms=train_transforms, heatmap_size=HEATMAP_SIZE,
                                   sigma=HEATMAP_SIGMA)
    temp_dataset.dataframe = temp_dataset.dataframe.reset_index(drop=True)

    # Filter out split-screen images from the supervised dataset
    if hasattr(temp_dataset, 'resolved_paths'):
        print("\n[+] Scanning and filtering split-screen images from the supervised dataset...")
        valid_indices = []
        valid_paths = []
        original_count = len(temp_dataset.resolved_paths)
        for i, path in enumerate(tqdm(temp_dataset.resolved_paths, desc="Filtering supervised")):
            if not is_split_screen(path):
                valid_indices.append(i)
                valid_paths.append(path)

        temp_dataset.dataframe = temp_dataset.dataframe.iloc[valid_indices].reset_index(drop=True)
        temp_dataset.resolved_paths = valid_paths

        filtered_sup_count = original_count - len(valid_paths)
        print(f"\n[+] Supervised data cleanup complete.")
        print(f"    - Initial supervised images: {original_count}")
        print(f"    - Filtered split-screen images: {filtered_sup_count}")
        print(f"    - Final supervised images for training: {len(valid_paths)}\n")

    task_configs = _build_task_configs(temp_dataset.dataframe)
    task_id_to_name = {cfg["task_id"]: cfg["task_name"] for cfg in task_configs}

    train_indices, val_indices = _stratified_split_indices(temp_dataset.dataframe, val_split=val_split,
                                                           seed=RANDOM_SEED)

    train_dataset = KeypointDataset(DATA_ROOT_PATH, transforms=train_transforms, heatmap_size=HEATMAP_SIZE,
                                    sigma=HEATMAP_SIGMA)
    train_dataset.dataframe = temp_dataset.dataframe.copy()
    val_dataset = KeypointDataset(DATA_ROOT_PATH, transforms=val_transforms, heatmap_size=HEATMAP_SIZE,
                                  sigma=HEATMAP_SIGMA)
    val_dataset.dataframe = temp_dataset.dataframe.copy()

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices].reset_index(drop=True)
    val_subset.dataframe = val_dataset.dataframe.iloc[val_indices].reset_index(drop=True)

    train_sampler = KeypointUniformSampler(train_subset, batch_size=BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_sampler=train_sampler, num_workers=4,
                                               pin_memory=True, collate_fn=keypoint_collate_fn,
                                               worker_init_fn=worker_init_fn, generator=g)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True,
                                             collate_fn=keypoint_collate_fn, worker_init_fn=worker_init_fn, generator=g)

    # Filter out split-screen images from the unlabeled dataset (Dynamic attribute detection)
    print("\n[+] Initializing unlabeled dataset and preparing for filtering...")
    unlabeled_dataset = UnlabeledDataset(DATA_ROOT_PATH, excluded_paths=temp_dataset.resolved_paths,
                                         transforms=train_transforms)

    path_attr_name = None
    for attr in ['samples', 'resolved_paths', 'image_paths', 'paths', 'img_paths', 'file_paths', 'unlabeled_paths',
                 'files', 'data']:
        if hasattr(unlabeled_dataset, attr) and isinstance(getattr(unlabeled_dataset, attr), list):
            if len(getattr(unlabeled_dataset, attr)) > 100:
                path_attr_name = attr
                break

    if path_attr_name is not None:
        original_paths = getattr(unlabeled_dataset, path_attr_name)
        print(f"[+] Found unlabeled path attribute `{path_attr_name}`. Scanning for split-screen images...")

        valid_unlabeled_paths = []
        valid_unlabeled_indices = []
        original_u_count = len(original_paths)

        for i, path in enumerate(tqdm(original_paths, desc="Filtering unlabeled")):
            if not is_split_screen(path):
                valid_unlabeled_paths.append(path)
                valid_unlabeled_indices.append(i)

        setattr(unlabeled_dataset, path_attr_name, valid_unlabeled_paths)

        if hasattr(unlabeled_dataset, 'dataframe') and unlabeled_dataset.dataframe is not None:
            try:
                unlabeled_dataset.dataframe = unlabeled_dataset.dataframe.iloc[valid_unlabeled_indices].reset_index(
                    drop=True)
            except Exception:
                pass

        filtered_unsup_count = original_u_count - len(valid_unlabeled_paths)
        print(f"\n[+] Unlabeled data cleanup complete.")
        print(f"    - Initial unlabeled images: {original_u_count}")
        print(f"    - Filtered split-screen images: {filtered_unsup_count}")
        print(f"    - Final unlabeled images for training: {len(valid_unlabeled_paths)}\n")
    else:
        print("\n[!] Warning: Could not detect the path attribute in UnlabeledDataset. Filtering skipped.")

    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=UNLABELED_BATCH_SIZE, shuffle=True,
                                                   drop_last=False, num_workers=4, pin_memory=True,
                                                   collate_fn=unlabeled_collate_fn, worker_init_fn=worker_init_fn,
                                                   generator=g)

    model = MultiTaskModelFactory(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, task_configs=task_configs,
                                  heatmap_size=HEATMAP_SIZE).to(device)
    ema_model = ModelEMA(model, decay=EMA_DECAY)

    param_groups = [{"params": model.encoder.parameters(), "lr": LEARNING_RATE * 0.2}]
    for task_id, head in model.heads.items():
        param_groups.append({"params": head.parameters(), "lr": LEARNING_RATE * 10.0})
    for task_id, dsnt_mod in model.dsnt_modules.items():
        param_groups.append({"params": dsnt_mod.parameters(), "lr": LEARNING_RATE * 10.0})

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    max_lrs = [group["lr"] for group in param_groups]
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lrs, steps_per_epoch=len(train_loader),
                                              epochs=NUM_EPOCHS, pct_start=0.3, anneal_strategy='cos')
    scaler = torch.cuda.amp.GradScaler()

    best_val_score = float("inf")
    start_epoch = 0

    # ================= Checkpoint Resumption =================
    if args.resume and os.path.exists(args.resume):
        print(f"\n[!] Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        ema_model.ema.load_state_dict(checkpoint['ema_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_score = checkpoint.get('best_val_score', float("inf"))
        print(f"[!] Resumed successfully. Starting from Epoch {start_epoch + 1}. Current best MRE: {best_val_score:.4f}\n")

    print(f"Checkpoints will be saved in: {CHECKPOINT_DIR}/")
    print("\n" + "=" * 50 + "\n--- Start Semi-Supervised Keypoint Training ---")

    # Initialize unlabeled iterator outside the epoch loop to ensure continuous traversal
    unlabeled_iter = iter(unlabeled_loader)

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")

        decay_factor = 1.0 - (epoch / float(NUM_EPOCHS))
        current_noise_std = (2.0 / INPUT_SIZE) * decay_factor
        consistency_weight = get_consistency_weight(epoch, NUM_EPOCHS, max_weight=UNSUP_MAX_WEIGHT)

        current_threshold = 0.5 + (0.35 * (epoch / NUM_EPOCHS))

        for batch_l in loop:
            try:
                batch_u = next(unlabeled_iter)
            except StopIteration:
                # Re-initialize the iterator once the unlabeled dataset is exhausted
                unlabeled_iter = iter(unlabeled_loader)
                batch_u = next(unlabeled_iter)

            images_l = batch_l["image"].to(device)
            task_ids_l = batch_l["task_id"]

            images_u = batch_u["image"].to(device)
            task_ids_u = batch_u["task_id"]

            optimizer.zero_grad(set_to_none=True)
            batch_loss_values = []

            # --- Step 1: Supervised Loss Calculation ---
            total_samples_in_batch_l = len(task_ids_l)
            for current_task_id in sorted(set(task_ids_l)):
                task_indices = [i for i, tid in enumerate(task_ids_l) if tid == current_task_id]
                task_images = images_l[task_indices]
                task_heatmaps = torch.stack([batch_l["heatmap"][i] for i in task_indices], 0).to(device)
                task_labels = torch.stack([batch_l["label"][i] for i in task_indices], 0).to(device)

                # Fetch task-specific multiplier
                t_weight = TASK_WEIGHTS.get(current_task_id, 1.0)

                B = task_labels.shape[0]
                K = task_labels.shape[1] // 2
                valid_mask = (task_labels.view(B, K, 2).sum(dim=-1) > 0).float()
                valid_mask_coords = valid_mask.unsqueeze(-1).repeat(1, 1, 2).view(B, K * 2)

                noise = torch.randn_like(task_labels) * current_noise_std
                task_labels_noisy = torch.clamp(task_labels + noise, 0.0, 1.0)

                with torch.cuda.amp.autocast():
                    pred_logits = model(task_images, task_id=current_task_id, return_coords=False)
                    pred_hm_sigmoid = torch.sigmoid(pred_logits)

                    weight_map = task_heatmaps * 9.0 + 1.0
                    loss_hm = (F.mse_loss(pred_hm_sigmoid, task_heatmaps, reduction='none') * weight_map).mean()

                    pred_coords_dsnt, _ = model.dsnt_modules[current_task_id](pred_logits)
                    pred_coords_01 = (pred_coords_dsnt + 1.0) / 2.0
                    pred_coords_01 = pred_coords_01.view(pred_coords_01.shape[0], -1)

                    # Coordinate loss changed to MSE (L2) to heavily penalize severe outliers and correct global spatial errors
                    loss_coord_unreduced = F.mse_loss(pred_coords_01, task_labels_noisy, reduction='none')
                    loss_coord = (loss_coord_unreduced * valid_mask_coords).sum() / (valid_mask_coords.sum() + 1e-8)

                    # MSE generally yields smaller values than L1, so the multiplier is elevated (50.0)
                    # to maintain gradient magnitude alongside task-specific weighting.
                    loss_supervised = loss_hm + 50.0 * loss_coord

                    # Apply task difficulty weight
                    weighted_loss = loss_supervised * (len(task_indices) / total_samples_in_batch_l) * t_weight

                scaler.scale(weighted_loss).backward()
                loss_value = float(loss_supervised.item())
                batch_loss_values.append(loss_value)
                epoch_train_losses[current_task_id].append(loss_value)

            # --- Step 2: Unsupervised Consistency Loss Calculation ---
            if consistency_weight > 0.0:
                total_samples_in_batch_u = len(task_ids_u)
                for current_task_id in sorted(set(task_ids_u)):
                    task_indices_u = [i for i, tid in enumerate(task_ids_u) if tid == current_task_id]
                    if not task_indices_u: continue

                    # Apply an inverse task weight for unsupervised data to prevent reinforcing early pseudo-label errors on hard tasks
                    t_weight_unsup = 1.0 / TASK_WEIGHTS.get(current_task_id, 1.0)
                    task_images_u = images_u[task_indices_u]

                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            teacher_logits = ema_model.ema(task_images_u, task_id=current_task_id, return_coords=False)
                            teacher_hm = torch.sigmoid(teacher_logits).detach()

                        B_u = teacher_hm.shape[0]
                        K_u = teacher_hm.shape[1]
                        peak_confs = teacher_hm.view(B_u, K_u, -1).max(dim=-1)[0]

                        conf_mask_hm = (peak_confs > current_threshold).float().view(B_u, K_u, 1, 1)

                        student_logits = model(task_images_u, task_id=current_task_id, return_coords=False)
                        student_hm = torch.sigmoid(student_logits)

                        loss_unsup_hm = (F.mse_loss(student_hm, teacher_hm, reduction='none') * conf_mask_hm).mean()

                        weighted_unsup_loss = loss_unsup_hm * consistency_weight * (
                                len(task_indices_u) / total_samples_in_batch_u) * t_weight_unsup

                    scaler.scale(weighted_unsup_loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ema_model.update(model)

            mean_batch_loss = float(np.mean(batch_loss_values)) if batch_loss_values else 0.0
            current_lr = scheduler.get_last_lr()[0]
            loop.set_postfix(loss=mean_batch_loss, unsup_w=consistency_weight, conf_thr=current_threshold,
                             lr=current_lr)

        print(f"\n--- Epoch {epoch + 1} Average Train Loss ---")
        for task_id in sorted(epoch_train_losses.keys()):
            avg_loss = float(np.mean(epoch_train_losses[task_id]))
            print(f"  - {task_id}: {avg_loss:.4f}")

        val_results_df = evaluate_keypoint(ema_model.ema, val_loader, device, task_id_to_name)
        selected_val_score = float("inf")
        if not val_results_df.empty and metric_column in val_results_df.columns:
            selected_val_score = float(val_results_df[metric_column].mean())

        print(f"\n--- Epoch {epoch + 1} Validation Report (Evaluated on EMA Model) ---")
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Val {metric_label} (Lower is better): {selected_val_score:.4f} ---")

        is_best = False
        if selected_val_score < best_val_score:
            best_val_score = selected_val_score
            is_best = True
            epoch_specific_save_path = os.path.join(CHECKPOINT_DIR,
                                                    f"best_epoch_{epoch + 1}_mre_{best_val_score:.4f}.pth")
            torch.save(ema_model.ema.state_dict(), epoch_specific_save_path)
            torch.save(ema_model.ema.state_dict(), MODEL_SAVE_PATH)
            print(f"-> New best model saved! {metric_label} improved to: {best_val_score:.4f}")

        logger.log_epoch(epoch=epoch + 1, train_losses_dict=epoch_train_losses, val_df=val_results_df,
                         avg_val_score=selected_val_score, metric_label=metric_label, is_best=is_best)

        # ================= Save Training State for Resumption =================
        training_state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'ema_state': ema_model.ema.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_val_score': best_val_score
        }
        torch.save(training_state, LATEST_STATE_PATH)

    print(f"\n--- Training Finished ---\nBest overall model saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-supervised keypoint training script")
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT, help="Validation split ratio")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to the latest_training_state.pth to resume training")
    args = parser.parse_args()
    main(args, val_split=float(args.val_split))