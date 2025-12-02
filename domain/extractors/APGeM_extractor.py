from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

logger = logging.getLogger(__name__)


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM)
    - AP-GeM, DELG 등에서 쓰이는 GeM의 기본 형태
    - p를 learnable parameter로 두어 다양한 pooling(평균~max 사이)을 표현
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = x.clamp(min=self.eps)
        x = x ** self.p
        # 전체 공간에 대해 avg_pool2d
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        x = x ** (1.0 / self.p)
        return x  # (B, C, 1, 1)


class APGeMDescriptorExtractor:
    """
    AP-GeM 계열 글로벌 디스크립터 추출기

    - backbone: tf_efficientnet_b3_ns (ImageNet pretrained)
    - pooling: GeM (learnable p)
    - 출력: L2-normalized descriptor (np.ndarray, float32)

    기존 GlobalDescriptorExtractor 와 동일하게:
        extract_one(image_path: Path) -> Optional[np.ndarray]
    인터페이스만 맞춰주면 DeepCluster 에 그대로 꽂을 수 있음.
    """

    def __init__(
        self,
        model_name: str = "tf_efficientnet_b3_ns",
        image_size: int = 320,
        device: Optional[str] = None,
    ) -> None:
        # -----------------------------
        # 1) device 선택 (mps -> cuda -> cpu)
        # -----------------------------
        if device is not None:
            self.device = torch.device(device)
        else:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        logger.info(f"[APGeM] using device = {self.device}")

        # -----------------------------
        # 2) backbone 생성 (분류 헤더 제거)
        #    - num_classes=0 → 마지막 FC 헤더 제거
        #    - global_pool='' → 우리가 직접 GeM을 붙이기 위해 풀링 제거
        # -----------------------------
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,   # feature extractor 모드
            global_pool="",  # 마지막 풀링 제거, (B, C, H, W) 그대로 받기
        )
        self.backbone.to(self.device)
        self.backbone.eval()

        # backbone 출력 채널 수 (descriptor 차원)
        self.feat_dim = self.backbone.num_features  # timm 모델에 공통 제공 필드 [oai_citation:2‡GitHub](https://github.com/huggingface/pytorch-image-models/discussions/1280?utm_source=chatgpt.com)
        logger.info(f"[APGeM] backbone={model_name}, feat_dim={self.feat_dim}")

        # -----------------------------
        # 3) GeM pooling + (원하면 FC 등 후처리 추가 가능)
        # -----------------------------
        self.pool = GeM(p=3.0).to(self.device)

        # -----------------------------
        # 4) 입력 전처리 (ImageNet norm)
        # -----------------------------
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),  # 보통 256→224 비율 비슷하게
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN,
                    std=IMAGENET_DEFAULT_STD,
                ),
            ]
        )

    @torch.no_grad()
    def extract_one(self, image_path: Path) -> Optional[np.ndarray]:
        """
        단일 이미지에 대한 L2-normalized AP-GeM descriptor 반환.
        실패 시 None.
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"[APGeM] Failed to open image {image_path}: {e}")
            return None

        x = self.transform(img).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # 1) backbone 통과 → feature map (1, C, h, w)
        feat_map = self.backbone(x)

        # EfficientNet 계열은 보통 (B, C, H, W)로 나옴
        if feat_map.ndim == 2:
            # 혹시 (B, C)로 나오는 모델이면, 이미 풀링된 상태일 수 있으니 그대로 사용
            desc = feat_map
        else:
            # 2) GeM pooling → (1, C, 1, 1)
            pooled = self.pool(feat_map)          # (1, C, 1, 1)
            desc = pooled.flatten(1)              # (1, C)

        # 3) L2 normalize
        desc = F.normalize(desc, p=2, dim=1)      # (1, C)

        # 4) numpy 로 변환
        return (
            desc.squeeze(0)                       # (C,)
            .cpu()
            .numpy()
            .astype(np.float32)
        )