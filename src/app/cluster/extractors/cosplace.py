import logging
from typing import List, Optional
import os

import numpy as np
from PIL import Image
from pyproj import Geod

# 도메인 의존성 (환경에 맞게 유지)
try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import torch
    from torchvision import transforms

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class CosPlaceExtractor:
    """
    ONNX Runtime을 우선 사용하여 CPU 추론 속도를 최적화한 Extractor.
    ONNX 모델이 없거나 로드 실패 시 PyTorch로 Fallback 합니다.
    """

    _session = None
    _torch_model = None
    _torch_preprocess = None

    # CosPlace ResNet50 Output Dim
    OUTPUT_DIM = 512
    # ImageNet Mean/Std for Numpy Preprocessing
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, onnx_path="img_models/cosplace_resnet50_int8.onnx"):
        self.use_onnx = False

        # 1. Try Loading ONNX
        if HAS_ONNX and os.path.exists(onnx_path):
            if CosPlaceExtractor._session is None:
                self._load_onnx_model(onnx_path)
            if CosPlaceExtractor._session is not None:
                self.use_onnx = True

        # 2. Fallback to Torch if ONNX is not available
        if not self.use_onnx:
            if HAS_TORCH:
                if CosPlaceExtractor._torch_model is None:
                    self._load_torch_model()
            else:
                logger.warning("Neither ONNX nor Torch is available. Feature extraction disabled.")

    @classmethod
    def _load_onnx_model(cls, path):
        logger.info(f"Loading ONNX model from {path}...")
        try:
            # CPU 전용 설정 (필요시 병렬 실행 옵션 조정 가능)
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2  # 코어 수에 맞게 조정
            sess_options.log_severity_level = 3  # Suppress warnings (like Unknown CPU vendor)
            
            # [Memory Optimization] Disable Memory Arena
            # This releases memory back to OS immediately after inference,
            # preventing huge memory retention when processing large batches.
            sess_options.enable_cpu_mem_arena = False
            
            cls._session = ort.InferenceSession(path, sess_options, providers=["CPUExecutionProvider"])
            logger.info("CosPlace ONNX model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            cls._session = None

    @classmethod
    def _load_torch_model(cls):
        logger.info("Loading CosPlace model from Torch Hub (Fallback)...")
        try:
            cls._torch_model = torch.hub.load(
                "gmberton/CosPlace",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=cls.OUTPUT_DIM,
                trust_repo=True,
            )
            cls._torch_model.eval()
            cls._torch_preprocess = transforms.Compose(
                [
                    transforms.Resize((480, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            logger.info("CosPlace Torch model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Torch model: {e}")

    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        [핵심 최적화] 이미지 배치를 한 번에 처리하여 Feature Matrix 반환
        Return: (Batch_Size, 512) Numpy Array
        """
        if not images:
            return np.array([])

        # A. ONNX Inference (Fastest on CPU)
        if self.use_onnx and self._session:
            try:
                # 1. Preprocess (Numpy Vectorization)
                # [Memory Optimization] Pre-allocate input tensor
                batch_size = len(images)
                input_tensor = np.zeros((batch_size, 3, 480, 640), dtype=np.float32)

                for i, img in enumerate(images):
                    img = img.convert("RGB").resize((640, 480))  # PIL uses (W, H)
                    img_np = np.array(img).astype(np.float32) / 255.0
                    img_np = (img_np - self.MEAN) / self.STD
                    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
                    input_tensor[i] = img_np

                # 2. Run Inference
                input_name = self._session.get_inputs()[0].name
                features = self._session.run(None, {input_name: input_tensor})[0]

                # 3. L2 Normalize
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                return features / (norms + 1e-6)

            except Exception as e:
                logger.error(f"ONNX Batch extraction failed: {e}")
                return np.zeros((len(images), self.OUTPUT_DIM))

        # B. Torch Inference (Fallback)
        elif HAS_TORCH and self._torch_model:
            try:
                batch_tensors = []
                device = next(self._torch_model.parameters()).device
                for img in images:
                    img = img.convert("RGB")
                    batch_tensors.append(self._torch_preprocess(img))

                input_batch = torch.stack(batch_tensors).to(device)

                with torch.no_grad():
                    features = self._torch_model(input_batch)
                    features = features.cpu().numpy()

                # L2 Normalize
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                return features / (norms + 1e-6)

            except Exception as e:
                logger.error(f"Torch Batch extraction failed: {e}")
                return np.zeros((len(images), self.OUTPUT_DIM))

        else:
            return np.zeros((len(images), self.OUTPUT_DIM))
