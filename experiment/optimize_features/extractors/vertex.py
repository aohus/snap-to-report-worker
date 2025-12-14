import base64
import io
import logging
import os
from typing import Optional, Union

import numpy as np
from google.cloud import aiplatform
from google.protobuf import struct_pb2
from PIL import Image, ImageFile

from experiment.optimize_features.extractors.base import BaseFeatureExtractor

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VertexExtractor(BaseFeatureExtractor):
    def __init__(self, project_id: Optional[str] = None, location: str = "asia-northeast3"):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "snap-2-report")
        self.location = location
        
        # Initialize Vertex AI
        try:
            aiplatform.init(project=self.project_id, location=self.location)
            self.client = aiplatform.gapic.PredictionServiceClient(
                client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
            )
            self.endpoint = f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/multimodalembedding@001"
            logger.info(f"VertexExtractor initialized for project {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise e

    def extract(self, image_input: Union[str, Image.Image]) -> Optional[np.ndarray]:
        """
        Extracts image embeddings using Google Vertex AI Multimodal Embedding API.
        
        Args:
            image_input (Union[str, Image.Image]): File path string OR PIL Image object (from masking)
        
        Returns:
            Optional[np.ndarray]: Normalized embedding vector or None if failed.
        """
        try:
            img = None
            should_close = False

            if isinstance(image_input, str):
                try:
                    img = Image.open(image_input)
                    should_close = True 
                except Exception as e:
                    logger.warning(f"Could not open image file {image_input}: {e}")
                    return None
            elif isinstance(image_input, Image.Image):
                img = image_input
                should_close = False
            else:
                logger.warning(f"Unsupported input type for VertexExtractor: {type(image_input)}")
                return None

            if img is None:
                return None

            try:
                img = img.convert("RGB")
                img.thumbnail((800, 800))
                output = io.BytesIO()
                img.save(output, format="JPEG", quality=85)
                resized_bytes = output.getvalue()
            finally:
                if should_close and img:
                    img.close()

            encoded_content = base64.b64encode(resized_bytes).decode("utf-8")
            instance = struct_pb2.Struct()
            instance.update({"image": {"bytesBase64Encoded": encoded_content}})
            
            instances = [instance]
            
            response = self.client.predict(endpoint=self.endpoint, instances=instances)
            embedding = response.predictions[0]['imageEmbedding']
            vector = np.array(embedding, dtype=np.float32)
            
            # 정규화 (Normalization) - 코사인 유사도 계산을 위해 필수
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
                
            return vector

        except Exception as e:
            source = image_input if isinstance(image_input, str) else "PIL Image Object"
            logger.warning(f"Vertex extraction failed for {source}: {e}")
            return None

    @property
    def dim(self) -> int:
        return 1408