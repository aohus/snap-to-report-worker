import logging
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

# 추가된 라이브러리 (설치 필요: pip install roboflow ultralytics opencv-python)
from roboflow import Roboflow
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class RoboflowMasker:
    def __init__(self):
        """
        Roboflow에서 학습된 YOLO 모델을 다운로드하고 로드합니다.
        """
        self.model = None
        try:
            model_path = "/Users/aohus/Workspaces/github/job-report-creator/cluster-backend/experiment/masking/runs/detect/train/weights/best.pt"
            print(f"Loading YOLO model from: {model_path}")
            self.model = YOLO(model_path)
            
        except Exception as e:
            logger.error(f"Failed to initialize RoboflowMasker: {e}")
            raise e

    def mask_image(self, image_path: str) -> Optional[Image.Image]:
        """
        이미지를 로드하고, 탐지된 모든 객체(작업자, 트럭, 사다리 등)를
        검은색 박스로 가린 뒤 PIL Image 객체로 반환합니다.
        """
        try:
            # OpenCV로 이미지 로드
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                logger.warning(f"Could not read image: {image_path}")
                return None

            # 객체 탐지 (conf=0.25: 확신도 25% 이상이면 다 잡기)
            # verbose=False로 로그 끄기
            results = self.model(img_cv, verbose=False, conf=0.40)

            # 탐지된 영역 블랙 마스킹
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                for box in boxes:
                    x1, y1, x2, y2 = box
                    # 검은색(0,0,0)으로 칠하기 (-1은 내부 채움)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 0), -1)

            # BGR -> RGB 변환 및 PIL Image로 변경
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
            
        except Exception as e:
            logger.error(f"Error masking image {image_path}: {e}")
            return None


if __name__ == "__main__":
    MY_API_KEY = "W5jHnLnencpZhLjEhEGr"
    PROJECT_NAME = "mask-noise"
    VERSION = 1

    # 2. 마스커 초기화 (자동으로 모델 다운로드 됨)
    masker = RoboflowMasker(N)

    # 3. 테스트
    masked_img = masker.mask_image("/Users/aohus/Workspaces/github/job-report-creator/cluster-backend/work_ye/0-label-1.jpeg")
    masked_img.show() # 검은 박스가 잘 쳐졌는지 확인