from typing import Optional
import numpy as np
from experiment.optimize_features.extractors.base import BaseFeatureExtractor

class NullExtractor(BaseFeatureExtractor):
    """
    Extractor that returns None. 
    Used for baselines (e.g., GPS + Time only) where no image features are needed.
    """
    def extract(self, image_path: str) -> Optional[np.ndarray]:
        # Return empty array so dataset.py considers it "valid" extraction
        # but effectively has no content.
        return np.array([])

    @property
    def dim(self) -> int:
        return 0
