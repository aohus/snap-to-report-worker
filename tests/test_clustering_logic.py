import sys
import os
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from unittest.mock import MagicMock

# --- Mock DB layer before imports ---
db_mock = MagicMock()
db_mock.AsyncSessionLocal = MagicMock()
db_mock.Base = MagicMock
sys.modules["app.db"] = MagicMock()
sys.modules["app.db.database"] = db_mock
# ------------------------------------

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from app.cluster.services.clustering import ClusteringService
from app.cluster.schema import ClusterRequest
from app.config import JobConfig
from app.common.models import Photo

@pytest.fixture
def callback_sender():
    return AsyncMock()

@pytest.fixture
def storage_mock():
    mock = MagicMock()
    mock.list_files = AsyncMock(return_value=["photo1.jpg", "photo2.jpg"])
    return mock

class DummyLocalStorage:
    pass

@pytest.mark.asyncio
async def test_clustering_process_task(callback_sender):
    # Mocking external interactions
    with patch("app.cluster.services.clustering.get_storage_client") as get_storage_mock, \
         patch("app.cluster.services.clustering.LocalStorageService", DummyLocalStorage), \
         patch("app.cluster.services.clustering.PhotoClusteringPipeline") as PipelineMock:
        
        # Setup Storage Mock
        storage_instance = MagicMock()
        storage_instance.list_files = AsyncMock(return_value=["photo1.jpg", "photo2.jpg"])
        storage_instance.media_root = Path("/tmp/media")
        storage_instance.get_url = lambda x: f"http://localhost/{x}"
        get_storage_mock.return_value = storage_instance
        
        # Make isinstance(storage, LocalStorageService) return False (Simulate GCS/S3 path)
        # OR return True to test local path logic
        # Let's test non-local logic first (simpler)
        # We need to ensure isinstance works. Mocking the class usually works if imported names match.
        # But verify service uses imported class.
        
        # Mock Pipeline
        pipeline_instance = AsyncMock()
        # Mock run return value
        # It expects List[List[PhotoMeta]] ?
        # pipeline.run returns "final_clusters". 
        # formatters.format_cluster_response expects list of clusters (list of photos?)
        # Let's check formatters logic if possible, or Mock format_cluster_response
        pipeline_instance.run.return_value = [
            # Cluster 1 (Valid, size 2)
            [
                MagicMock(original_name="photo1.jpg", path="photos/photo1.jpg", timestamp=1234567890.0, lat=37.0, lon=127.0),
                MagicMock(original_name="photo1_2.jpg", path="photos/photo1_2.jpg", timestamp=1234567895.0, lat=37.0, lon=127.0),
            ],
            # Cluster 2 (Noise, size 1)
             [
                MagicMock(original_name="photo2.jpg", path="photos/photo2.jpg", timestamp=None, lat=None, lon=None),
            ]
        ]
        PipelineMock.return_value = pipeline_instance

        service = ClusteringService(callback_sender)
        
        req = ClusterRequest(
            webhook_url="http://callback.com",
            bucket_path="user/job/photos",
            min_samples=2,
            request_id="req-123",
            photo_cnt=2,
            cluster_job_id="job-123"
        )
        
        lock = asyncio.Lock()
        
        await service.process_task("task-1", req, lock)
        
        # Verify Interactions
        get_storage_mock.assert_called_once()
        storage_instance.list_files.assert_called_once_with("user/job/photos")
        
        # Verify Pipeline Init
        # Should be called with JobConfig, Storage, Photos
        PipelineMock.assert_called_once()
        args, kwargs = PipelineMock.call_args
        assert isinstance(args[0], JobConfig)
        assert args[0].job_id == "job-123"
        assert len(args[2]) == 2 # 2 photos
        
        # Verify Pipeline Run
        pipeline_instance.run.assert_called_once()
        
        # Verify Callback
        callback_sender.send_result.assert_called_once()
        call_args = callback_sender.send_result.call_args
        assert call_args[0][0] == "http://callback.com"
        assert call_args[0][2] == "task-1"
        payload = call_args[0][1]
        assert payload["status"] == "completed"
        assert payload["result"]["total_clusters"] == 2
