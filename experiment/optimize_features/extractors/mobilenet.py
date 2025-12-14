import logging
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from experiment.optimize_features.extractors.base import BaseFeatureExtractor

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MobileNetExtractor(BaseFeatureExtractor):
    _instance = None
    _model = None
    _preprocess = None

    def __init__(self):
        if MobileNetExtractor._model is None:
            self._load_model()

    @classmethod
    def _load_model(cls):
        logger.info("Loading MobileNetV3-Small model...")
        weights = MobileNet_V3_Small_Weights.DEFAULT
        cls._model = mobilenet_v3_small(weights=weights)
        cls._model.eval()
        
        # 마지막 분류 레이어(classifier)를 제거하여 Feature Vector만 뽑도록 설정
        # (원래는 1000개 클래스 확률이 나오지만, 이걸 Identity로 바꾸면 직전 벡터가 나옴)
        # MobileNetV3 Small의 마지막 feature dim은 576입니다.
        cls._model.classifier = torch.nn.Identity()
        
        cls._preprocess = weights.transforms()

    def extract(self, image_input: Union[str, Image.Image]) -> Optional[np.ndarray]:
        """
        이미지 경로(str) 또는 PIL Image 객체를 받아 Feature Vector를 반환
        """
        img = None
        should_close = False
        
        try:
            if isinstance(image_input, str):
                img = Image.open(image_input)
                should_close = True
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                logger.error(f"Unsupported input type for MobileNet: {type(image_input)}")
                return None

            # MobileNet 전처리기가 리사이징, 크롭, 정규화를 알아서 수행함
            img = img.convert("RGB")
            input_tensor = self._preprocess(img).unsqueeze(0)
            
            # 추론 (Inference)
            with torch.no_grad():
                feature_vector = self._model(input_tensor)
            vector = feature_vector.squeeze().numpy()
            
            # 코사인 유사도(np.dot) 사용을 위해 L2 정규화 수행
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
                
            return vector

        except Exception as e:
            source = image_input if isinstance(image_input, str) else "PIL Image Object"
            logger.error(f"MobileNet extraction failed for {source}: {e}")
            return None
            
        finally:
            if should_close and img:
                img.close()

    @property
    def dim(self) -> int:
        return 576