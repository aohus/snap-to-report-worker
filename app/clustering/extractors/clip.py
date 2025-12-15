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


class CLIPDescriptorExtractor(BaseDescriptorExtractor):
    def __init__(
        self,
        config: ClusteringConfig,
        people_detector: Optional[PeopleDetector] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device if device is not None else DEVICE
        self.config = config.clip
        self.people_detector = people_detector

        import open_clip  # Lazy import
        self.model, _, preprocess = open_clip.create_model_and_transforms(
            self.config.model_name, pretrained=self.config.pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess
        logger.info(f"[CLIP] model={self.config.model_name}, pretrained={self.config.pretrained}, device={self.device}")


    @torch.no_grad()
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"[CLIP] Failed to open image {image_path}: {e}")
            return None

        if self.people_detector:
            img = self.people_detector.mask_people_in_image(img)

        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
        feat_np = features.cpu().numpy().flatten().astype(np.float32)
        return feat_np

