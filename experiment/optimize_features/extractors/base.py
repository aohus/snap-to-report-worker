from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    @abstractmethod
    def extract(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extracts a feature vector from an image.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            Optional[np.ndarray]: Feature vector or None if extraction fails.
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Returns the dimension of the feature vector."""
        pass
