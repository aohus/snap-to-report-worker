import logging
import os
import pickle
from typing import Dict, List, Optional, Union

# Ensure app context is available if needed, or mocking PhotoMeta
try:
    from app.common.models import PhotoMeta
except ImportError:
    # Fallback if running outside of app context
    from dataclasses import dataclass
    @dataclass
    class PhotoMeta:
        path: str
        lat: Optional[float] = None
        lon: Optional[float] = None
        timestamp: Optional[float] = None
        # Dynamic attribute for label_id will be added at runtime

from experiment.optimize_features.extractors import BaseFeatureExtractor
from experiment.optimize_features.masking import RoboflowMasker

logger = logging.getLogger(__name__)


def prepare_dataset(
    extractor: BaseFeatureExtractor, 
    photos: List[PhotoMeta], 
    output_path: str,
    masker: Optional[RoboflowMasker] = None  # 마스커 인자 추가
):
    """
    Extracts features for the given photos using the specified extractor and saves them to a pickle file.
    If a 'masker' is provided, it masks objects before extraction.
    """
    features = []
    valid_photos = []
    
    extractor_name = extractor.__class__.__name__
    print(f"Extracting features using {extractor_name}...")
    if masker:
        print("-> Object Masking is ENABLED (Removing workers/vehicles...)")

    for i, p in enumerate(photos):
        try:
            if masker:
                input_data = masker.mask_image(p.path)
                if input_data is None:
                    continue
            else:
                input_data = p.path
            vector = extractor.extract(input_data)
            if vector is not None:
                features.append(vector)
                valid_photos.append(p)
            else:
                logger.warning(f"Failed to extract feature for {p.path}")
                
        except Exception as e:
            logger.error(f"Error processing {p.path}: {e}")
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(photos)}")

    data = {
        "photos": valid_photos,
        "features": features,
        "extractor_name": extractor_name,
        "extractor_dim": extractor.dim,
        "masked": masker is not None
    }

    # 디렉토리 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved {len(features)} features to {output_path}")


def load_dataset(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    try:
        masker = RoboflowMasker()
    except Exception as e:
        print(f"Masker init failed: {e}. Proceeding without masking.")
        masker = None

    # 3. 데이터셋 준비 실행 (예시)
    # dummy_extractor = ... (여기에 실제 extractor 객체 필요)
    # dummy_photos = [PhotoMeta(path="test.jpg"), ...]
    # prepare_dataset(dummy_extractor, dummy_photos, "output/features.pkl", masker=masker)