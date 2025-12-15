from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from clustering.extractors.apgem import APGeMDescriptorExtractor
from clustering.extractors.base_extractor import BaseDescriptorExtractor
from clustering.extractors.clip import CLIPDescriptorExtractor
from clustering.extractors.people_detector import PeopleDetector
from config import ClusteringConfig
from core.device import DEVICE
from PIL import Image, ImageFile

logger = logging.getLogger(__name__)

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CombinedAPGeMCLIPExtractor(BaseDescriptorExtractor):
    def __init__(
        self,
        config: ClusteringConfig,
        people_detector: Optional[PeopleDetector] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device if device is not None else DEVICE
        self.descriptor_config = config.descriptor
        
        self.apgem = APGeMDescriptorExtractor(config, people_detector=people_detector, device=self.device)
        self.clip = CLIPDescriptorExtractor(config, people_detector=people_detector, device=self.device)
        self.w_apgem = self.descriptor_config.w_apgem
        self.w_clip = self.descriptor_config.w_clip
        # The experiment code had l2_normalize_final, but for combined features, it's generally good practice to always normalize.
        self.l2_normalize_final = True 

    @torch.no_grad()
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        f_ap = self.apgem.extract_one(image_path)
        f_cl = self.clip.extract_one(image_path)

        if f_ap is None and f_cl is None:
            return None

        parts: list[np.ndarray] = []

        if f_ap is not None:
            fa = f_ap.astype(np.float32)
            fa = fa / (np.linalg.norm(fa) + 1e-8)
            fa = fa * self.w_apgem
            parts.append(fa)

        if f_cl is not None:
            fc = f_cl.astype(np.float32)
            fc = fc / (np.linalg.norm(fc) + 1e-8)
            fc = fc * self.w_clip
            parts.append(fc)

        if not parts:
            return None

        combined = np.concatenate(parts).astype(np.float32)

        if self.l2_normalize_final:
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm

        return combined