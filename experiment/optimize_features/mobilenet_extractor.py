import io
import logging
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from pyproj import Geod
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class FeatureExtractor:
    _instance = None
    _model = None
    _preprocess = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # CPU 모드로 경량 모델 로드
            # pretrained=True: 이미지넷 데이터로 학습된 가중치 사용 (사물의 특징을 잘 앎)
            weights = MobileNet_V3_Small_Weights.DEFAULT
            cls._model = mobilenet_v3_small(weights=weights)
            cls._model.eval() # 평가 모드 (학습 X)
            
            # 마지막 분류 레이어(Classifier) 제거 -> 특징 벡터(Embedding)만 추출
            # MobileNetV3 Small의 마지막 features 출력은 576차원
            cls._model.classifier = torch.nn.Identity()
            
            cls._preprocess = weights.transforms()
        return cls._model, cls._preprocess


def _extract_mobilenet_feature(path: str) -> Optional[np.ndarray]:
    """
    MobileNetV3를 사용하여 이미지의 의미론적 특징(Semantic Feature) 추출
    """
    try:
        img_data = None
        with open(path, "rb") as f: 
            img_data = f.read()

        if not img_data: 
            return None

        # 2. 전처리
        with Image.open(io.BytesIO(img_data)) as img:
            img = img.convert("RGB") # PyTorch 모델은 RGB 3채널 필요
            
            # 모델 로드 (싱글톤 패턴 활용)
            model, preprocess = FeatureExtractor.get_model()
            
            # 이미지 텐서 변환 및 정규화
            input_tensor = preprocess(img).unsqueeze(0) # Batch 차원 추가
            
            # 3. 추론 (Inference) - CPU
            with torch.no_grad():
                feature_vector = model(input_tensor)
            
            # (1, 576) -> (576,) numpy array
            return feature_vector.squeeze().numpy()

    except Exception as e:
        logger.error(f"MobileNet extraction failed: {e}")
        return None
    
