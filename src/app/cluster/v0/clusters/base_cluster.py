from abc import ABC, abstractmethod
from typing import Iterable

from models.photometa import PhotoMeta


class Clusterer(ABC):
    """Abstract base class for a clustering strategy."""

    @abstractmethod
    async def cluster(self, photos: list[PhotoMeta]) -> list[list[PhotoMeta]]:
        """
        Applies a clustering strategy to a list of photos.

        Args:
            photos: A list of PhotoMeta objects to cluster.

        Returns:
            A list of clusters, where each cluster is a list of PhotoMeta objects.
        """
        raise NotImplementedError()

    @staticmethod
    def condition(photos: Iterable[PhotoMeta]) -> bool:
        """Determines whether the strategy should be applied to the given photos."""
        return True
