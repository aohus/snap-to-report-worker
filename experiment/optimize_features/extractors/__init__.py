from experiment.optimize_features.extractors.base import BaseFeatureExtractor
from experiment.optimize_features.extractors.cosplace import CosPlaceExtractor
from experiment.optimize_features.extractors.mobilenet import MobileNetExtractor
from experiment.optimize_features.extractors.null import NullExtractor
from experiment.optimize_features.extractors.vertex import VertexExtractor


def get_extractor(name: str, **kwargs) -> BaseFeatureExtractor:
    name_lower = name.lower()
    if name_lower == "mobilenet":
        return MobileNetExtractor()
    elif name_lower == "vertex":
        return VertexExtractor(**kwargs)
    elif name_lower == "cosplace":
        return CosPlaceExtractor(**kwargs)
    elif name_lower in ["null", "gps_time", "none"]:
        return NullExtractor()
    else:
        raise ValueError(f"Unknown extractor: {name}")

# __all__ = [
#     "BaseFeatureExtractor",
#     "MobileNetExtractor",
#     "VertexExtractor",
#     "NetVLADExtractor",
#     "NullExtractor",
#     "get_extractor",
# ]