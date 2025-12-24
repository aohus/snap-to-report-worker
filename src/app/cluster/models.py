from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class PhotoMeta:
    id: str
    path: str
    original_name: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None
    timestamp: Optional[float] = None  # Unix timestamp
    focal_length: Optional[float] = None
    device: Optional[str] = None
    orientation: Optional[int] = None
    digital_zoom: Optional[float] = None
    scene_capture_type: Optional[int] = None
    white_balance: Optional[int] = None
    exposure_time: Optional[float] = None
    iso_speed_rating: Optional[int] = None
    flash: Optional[int] = None
    gps_img_direction: Optional[float] = None  # 방위각(도 단위)
    thumbnail_path: Optional[str] = None
    thumbnail_url: Optional[str] = None
    url: Optional[str] = None

@dataclass
class Photo:
    storage_path: str
    thumbnail_path: Optional[str] = None
    original_filename: Optional[str] = None
    meta_lat: Optional[float] = None
    meta_lon: Optional[float] = None
    meta_timestamp: Optional[datetime] = None 

