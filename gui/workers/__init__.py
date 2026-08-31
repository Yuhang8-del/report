"""Background workers used by the file-based desktop demonstration."""

from fruit_ssod.gui.workers.image_worker import (
    ImageInferenceResult,
    ImageInferenceSettings,
    ImageInferenceWorker,
)
from fruit_ssod.gui.workers.video_worker import (
    VideoFrameResult,
    VideoInferenceSettings,
    VideoInferenceWorker,
    VideoSourceInfo,
)

__all__ = [
    "ImageInferenceResult", "ImageInferenceSettings", "ImageInferenceWorker",
    "VideoFrameResult", "VideoInferenceSettings", "VideoInferenceWorker", "VideoSourceInfo",
]
