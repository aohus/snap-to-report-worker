import base64
import io
import logging
from typing import Optional

import numpy as np

# GCP Libraries
from google.cloud import aiplatform
from google.protobuf import struct_pb2
from PIL import Image, ImageFile

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class VertexEmbeddingClient:
    _instance = None
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
        # 멀티모달 임베딩 모델
        self.client = aiplatform.predicition.PredictionServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        )
        self.endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/multimodalembedding@001"

    @classmethod
    def get_instance(cls):
        # 실제 환경에 맞게 Project ID와 Region 설정 필요
        # 환경 변수에서 가져오도록 수정 권장
        if cls._instance is None:
            # TODO: 프로젝트 ID를 설정 파일이나 환경 변수에서 가져오세요.
            # 예: os.getenv("GOOGLE_CLOUD_PROJECT")
            cls._instance = cls("YOUR_PROJECT_ID", "us-central1") 
        return cls._instance

    def get_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        try:
            encoded_content = base64.b64encode(image_bytes).decode("utf-8")
            instance = struct_pb2.Struct()
            instance.update({"image": {"bytesBase64Encoded": encoded_content}})
            
            instances = [instance]
            # 텍스트 없이 이미지만 보냄 -> 1408차원 벡터 반환
            response = self.client.predict(endpoint=self.endpoint, instances=instances)
            
            embedding = response.predictions[0]['imageEmbedding']
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Vertex AI API Error: {e}")
            return None


def _extract_vertex_feature(path: str) -> Optional[np.ndarray]:
    """
    이미지 다운로드 -> 리사이즈 -> Vertex AI API 호출 -> 임베딩 반환
    """
    try:
        with open(path, "rb") as f:
            img_data = f.read()

        if not img_data: return None

        with Image.open(io.BytesIO(img_data)) as img:
            img = img.convert("RGB")
            # 긴 변 기준 800px 정도로 리사이즈 (디테일 유지하면서 용량 줄임)
            img.thumbnail((800, 800))
            
            # 다시 바이트로 변환
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=85)
            resized_bytes = output.getvalue()

        # 3. Vertex AI API 호출
        # 여기서 싱글톤 클라이언트 호출
        # (주의: 프로젝트 ID 설정 필요)
        ai_client = VertexEmbeddingClient.get_instance()
        vector = ai_client.get_embedding(resized_bytes)
        
        # 정규화 (Cosine Similarity 계산을 위해 미리 해두면 좋음)
        if vector is not None:
            norm = np.linalg.norm(vector)
            if norm > 0: vector /= norm
            
        return vector

    except Exception as e:
        logger.warning(f"Feature extraction failed for {path}: {e}")
        return None
