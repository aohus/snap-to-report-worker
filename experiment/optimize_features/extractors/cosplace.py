import logging

import numpy as np
import torch
from PIL import Image, ImageFile

from experiment.optimize_features.extractors.base import BaseFeatureExtractor

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CosPlaceExtractor(BaseFeatureExtractor):
    _model = None
    _preprocess = None
    
    # CosPlace는 출력 차원을 선택할 수 있습니다 (기본 512, 1024, 2048 등)
    # PCA가 없어도 차원이 작아 바로 쓰기 좋습니다.
    OUTPUT_DIM = 512 

    def __init__(self):
        if CosPlaceExtractor._model is None:
            self._load_model()

    @classmethod
    def _load_model(cls):
        logger.info("Loading CosPlace model from Torch Hub...")
        # 'gmberton/CosPlace' 리포지토리에서 자동으로 모델과 가중치를 가져옵니다.
        # backbone: ResNet50 (or ResNet18), fc_output_dim: 512
        cls._model = torch.hub.load("gmberton/CosPlace", "get_trained_model", 
                                    backbone="ResNet50", fc_output_dim=cls.OUTPUT_DIM)
        cls._model.eval()
        
        if torch.cuda.is_available():
            cls._model = cls._model.cuda()
            
        # CosPlace는 표준 ResNet 전처리를 사용하지 않고 자체적인 간단한 전처리를 권장할 때가 많으나,
        # 편의상 Torch Hub 모델 내부는 일반적인 ImageNet Normalize를 기대합니다.
        from torchvision import transforms
        cls._preprocess = transforms.Compose([
            transforms.Resize((480, 640)), # 장소 인식에 적합한 해상도
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image_input) -> np.ndarray:
        # (이전 코드와 동일한 이미지 로딩 로직)
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = image_input
        img = img.convert("RGB")

        device = next(self._model.parameters()).device
        input_tensor = self._preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # CosPlace 모델은 (N, D) 텐서를 반환
            feature = self._model(input_tensor)
            
        vector = feature.cpu().numpy().flatten()
        
        # L2 Normalize (CosPlace는 학습 시 되어있지만, 안전을 위해 한 번 더 수행)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
            
        return vector

    @property
    def dim(self) -> int:
        return self.OUTPUT_DIM