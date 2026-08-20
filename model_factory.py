from typing import Dict, List, Tuple, Union
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX", "FUGC", "fetal_femur"}


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.
    A lightweight channel attention mechanism that helps independent task heads
    filter critical features from the backbone while suppressing background noise.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid_channels, bias=False),
            nn.GELU(),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weight = self.fc(x).view(b, c, 1, 1)
        return x * weight


class DSNT(nn.Module):
    """
    Differentiable Spatial to Numerical Transform (DSNT).
    Each task head maintains its own instance with a learnable temperature
    parameter to prevent gradient interference between different tasks.
    """
    def __init__(self, heatmap_size=(64, 64)):
        super(DSNT, self).__init__()
        h, w = heatmap_size if isinstance(heatmap_size, tuple) else (heatmap_size, heatmap_size)

        x_coords = torch.linspace(-1, 1, w, dtype=torch.float32)
        y_coords = torch.linspace(-1, 1, h, dtype=torch.float32)

        self.grid_y, self.grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.register_buffer('grid_x_b', self.grid_x.view(1, 1, h, w), persistent=False)
        self.register_buffer('grid_y_b', self.grid_y.view(1, 1, h, w), persistent=False)

        # Task-specific adaptive sharpening coefficient (temperature)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_keypoints, height, width = heatmaps.shape
        reshaped_heatmaps = heatmaps.view(batch_size, num_keypoints, -1)

        # Spatial Softmax combined with task-specific temperature
        normalized_heatmaps = F.softmax(reshaped_heatmaps * self.temperature, dim=2)
        normalized_heatmaps = normalized_heatmaps.view(batch_size, num_keypoints, height, width)

        # Calculate spatial coordinate expectation (Center of Mass)
        coord_x = torch.sum(self.grid_x_b * normalized_heatmaps, dim=[2, 3])
        coord_y = torch.sum(self.grid_y_b * normalized_heatmaps, dim=[2, 3])

        coords = torch.stack((coord_x, coord_y), dim=2)
        return coords, normalized_heatmaps


class HeatmapHead(nn.Module):
    """
    Spatial-aware decoder designed to reduce checkerboard artifacts:
    1. Uses Bilinear Upsample + Conv2d instead of ConvTranspose2d.
    2. Injects coordinate grids after feature amplification for precise spatial priors.
    3. Integrates SEBlock for enhanced multi-task feature isolation.
    """
    def __init__(self, in_channels: int, num_points: int):
        super().__init__()
        hidden = max(in_channels // 2, 256)

        # First smooth upsampling block
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels + 2, hidden, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout2d(p=0.1)

        # Second smooth upsampling block
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(hidden + 2, hidden // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden // 2)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout2d(p=0.05)

        # High-frequency detail residual refinement block
        self.refine_conv1 = nn.Conv2d(hidden // 2 + 2, hidden // 2, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden // 2)
        self.act3 = nn.GELU()

        self.refine_conv2 = nn.Conv2d(hidden // 2, hidden // 2, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(hidden // 2)

        # Channel attention filtering
        self.se = SEBlock(hidden // 2, reduction=16)
        self.act4 = nn.GELU()

        # Final projection to keypoints
        self.final_proj = nn.Conv2d(hidden // 2, num_points, kernel_size=1)

    def _append_coordinate_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        """Dynamically generates and concatenates normalized coordinate grids [-1, 1]"""
        b, _, h, w = tensor.shape
        y_grid = torch.linspace(-1, 1, h, device=tensor.device, dtype=tensor.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        x_grid = torch.linspace(-1, 1, w, device=tensor.device, dtype=tensor.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([tensor, x_grid, y_grid], dim=1)

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        # Step 1: First upsampling and low-res coordinate injection
        x = self.upsample1(x)
        x = self._append_coordinate_grid(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        # Step 2: Second upsampling and mid-res coordinate injection
        x = self.upsample2(x)
        x = self._append_coordinate_grid(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        # Step 3: Residual refinement and channel attention
        x_coord = self._append_coordinate_grid(x)
        out = self.refine_conv1(x_coord)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.refine_conv2(out)
        out = self.bn4(out)

        # Fuse residual and apply channel attention weighting
        x = self.act4(self.se(x + out))

        # Step 4: Map to final keypoint heatmaps
        x = self.final_proj(x)

        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class DINOv2Backbone(nn.Module):
    """Extracts patch tokens from the last layer of ViT and reshapes them into 2D feature maps."""
    def __init__(self, model_name: str = "vit_small_patch14_dinov2.lvd142m", pretrained: bool = True):
        super().__init__()
        timm = importlib.import_module("timm")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        if not hasattr(self.backbone, "patch_embed"):
            raise ValueError(f"Model '{model_name}' is not a ViT-style backbone with patch_embed.")
        self.out_channels = int(self.backbone.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)

        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                patch_tokens = feats["x_norm_patchtokens"]
            elif "x_prenorm" in feats:
                all_tokens = feats["x_prenorm"]
                patch_tokens = all_tokens[:, 1:, :]
            else:
                raise RuntimeError("Unsupported forward_features output from DINOv2 backbone.")
        elif isinstance(feats, torch.Tensor):
            if feats.dim() == 3:
                patch_tokens = feats[:, 1:, :]
            else:
                raise RuntimeError("Unexpected tensor shape from forward_features.")
        else:
            raise RuntimeError("Unexpected feature type from DINOv2 backbone.")

        bsz, num_tokens, channels = patch_tokens.shape
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise RuntimeError("Patch token count is not square; input size may be incompatible.")

        feat_map = patch_tokens.transpose(1, 2).reshape(bsz, channels, side, side)
        return feat_map


class MultiTaskModelFactory(nn.Module):
    """Multi-task high-precision regression model factory."""
    def __init__(
            self,
            encoder_name: str,
            encoder_weights: str,
            task_configs: List[Dict],
            heatmap_size=(64, 64),
    ):
        super().__init__()
        self.heatmap_size = heatmap_size

        print(f"Initializing DINOv2 encoder: {encoder_name}")
        self.encoder = DINOv2Backbone(model_name=encoder_name, pretrained=(encoder_weights is not None))

        self.heads = nn.ModuleDict()
        # Individual DSNT modules assigned per task
        self.dsnt_modules = nn.ModuleDict()

        print(f"Creating keypoint heads and individual DSNT modules for {len(task_configs)} tasks...")

        for config in task_configs:
            task_id = config["task_id"]
            task_name = config["task_name"]
            if task_name != "Regression" and task_id not in EXTRA_REGRESSION_TASK_IDS:
                continue

            num_points = int(config["num_classes"])
            self.heads[task_id] = HeatmapHead(in_channels=self.encoder.out_channels, num_points=num_points)
            self.dsnt_modules[task_id] = DSNT(heatmap_size=self.heatmap_size)

        if not self.heads:
            raise ValueError("No keypoint heads were created. Check task_configs with task_name == 'Regression'.")

    def forward(self, x: torch.Tensor, task_id: str, return_coords: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the specified task.
        - return_coords=False: Returns raw heatmaps [B, K, H, W]
        - return_coords=True: Returns DSNT computed outputs (sub-pixel coordinates, normalized heatmaps)
        """
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found in keypoint heads.")

        features = self.encoder(x)
        raw_heatmaps = self.heads[task_id](features, out_size=self.heatmap_size)

        if not return_coords:
            return raw_heatmaps

        # Apply task-specific DSNT transformer
        coords, normalized_heatmaps = self.dsnt_modules[task_id](raw_heatmaps)
        return coords, normalized_heatmaps