import logging
import pickle

import numpy as np

from experiment.optimize_features.mobilenet_extractor import _extract_mobilenet_feature
from experiment.optimize_features.vertex_extractor import _extract_vertex_feature

logger = logging.getLogger(__name__)

def prepare_dataset():
    """
    정답셋 1000장의 임베딩을 미리 추출하여 저장합니다.
    """
    # 1. 정답셋 로딩 (예시: CSV나 DB에서 로드한다고 가정)
    # photos 리스트에는 각 사진의 정답 라벨(label_id)이 있어야 합니다.
    # photos: List[PhotoMeta] = load_ground_truth_photos()
    
    # 2. 임베딩 추출 (Vertex AI)
    # 비용 절감을 위해 여기서 한 번만 API를 호출합니다.
    # vertex_client = VertexEmbeddingClient.get_instance()
    
    features = []
    valid_photos = []
    
    print("Extracting embeddings for optimization...")
    for p in photos:
        # 로컬 이미지 로드 및 리사이즈 로직 필요 (이전 코드 참조)
        # img_bytes = load_and_resize(p.path)
        # vector = vertex_client.get_embedding(img_bytes)
        vector = _extract_mobilenet_feature(p.path)
        
        # 테스트용 더미 데이터 (실제 실행 시엔 위 로직 사용)
        vector = np.random.rand(1408).astype(np.float32) 
        
        if vector is not None:
            features.append(vector)
            valid_photos.append(p)
            
    # 3. 데이터 저장 (Pickle)
    # 나중에 Optuna가 이 파일을 계속 재사용합니다.
    with open("dataset_cache.pkl", "wb") as f:
        pickle.dump({
            "photos": valid_photos,
            "features": features
        }, f)
    
    print(f"Saved {len(features)} features to dataset_cache.pkl")

# 실행
# prepare_dataset()