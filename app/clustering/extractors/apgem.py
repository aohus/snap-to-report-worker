from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ClusteringConfig
from core.device import DEVICE
from PIL import Image, ImageFile
from services.clustering.base import BaseDescriptorExtractor
from services.clustering.people_detector import PeopleDetector  # New import
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

logger = logging.getLogger(__name__)

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps)
        x = x ** self.p
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        x = x ** (1.0 / self.p)
        return x


class APGeMDescriptorExtractor(BaseDescriptorExtractor):
    def __init__(
        self,
        config: ClusteringConfig,
        people_detector: Optional[PeopleDetector] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device if device is not None else DEVICE
        self.config = config.apgem
        self.people_detector = people_detector

        self.backbone = timm.create_model(
            self.config.model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        self.backbone.to(self.device)
        self.backbone.eval()

        self.feat_dim = self.backbone.num_features
        logger.info(f"[APGeM] backbone={self.config.model_name}, feat_dim={self.feat_dim}, device={self.device}")

        self.pool = GeM(p=3.0).to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(int(self.config.image_size * 1.14)),
                transforms.CenterCrop(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN,
                    std=IMAGENET_DEFAULT_STD,
                ),
            ]
        )



    @torch.no_grad()
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"[APGeM] Failed to open image {image_path}: {e}")
            return None
        
        if self.people_detector:
            img = self.people_detector.mask_people_in_image(img)

        x = self.transform(img).unsqueeze(0).to(self.device)
        feat_map = self.backbone(x)

        if feat_map.ndim == 2:
            desc = feat_map
        else:
            pooled = self.pool(feat_map)
            desc = pooled.flatten(1)

        desc = F.normalize(desc, p=2, dim=1)
        return (
            desc.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def __repr__(self) -> str:
        return f"APGeMDescriptorExtractor(model_name={self.config.model_name}, image_size={self.config.image_size}, device={self.device})"