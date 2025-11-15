"""Configuration module for face detection application."""
import os
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for application settings and paths."""
    
    def __init__(self) -> None:
        # Project root directory
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Paths to haar cascade XML files
        self.HAAR_CASCADE_DIR = os.path.join(self.PROJECT_ROOT, "haarcascades")
        self.FACE_CASCADE_PATH = os.path.join(
            self.HAAR_CASCADE_DIR, "haarcascade_frontalface_default.xml"
        )
        self.EYE_CASCADE_PATH = os.path.join(
            self.HAAR_CASCADE_DIR, "haarcascade_eye.xml"
        )

        # Fallback to OpenCV's built-in cascade files if local files don't exist
        if not os.path.exists(self.FACE_CASCADE_PATH):
            import cv2
            opencv_data_dir = cv2.data.haarcascades
            self.FACE_CASCADE_PATH = os.path.join(
                opencv_data_dir, "haarcascade_frontalface_default.xml"
            )
            logger.info(f"Using OpenCV built-in face cascade: {self.FACE_CASCADE_PATH}")
        
        if not os.path.exists(self.EYE_CASCADE_PATH):
            import cv2
            opencv_data_dir = cv2.data.haarcascades
            self.EYE_CASCADE_PATH = os.path.join(
                opencv_data_dir, "haarcascade_eye.xml"
            )
            logger.info(f"Using OpenCV built-in eye cascade: {self.EYE_CASCADE_PATH}")

        # Data directory paths
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, "data")
        self.IMAGES_DIR = os.path.join(self.DATA_DIR, "images")

        # Ensure directories exist
        os.makedirs(self.HAAR_CASCADE_DIR, exist_ok=True)
        os.makedirs(self.IMAGES_DIR, exist_ok=True)

        # Detection parameters
        self.MIN_NEIGHBORS = 5
        self.SCALE_FACTOR = 1.1

        # Colors for visualization (in BGR format for OpenCV)
        self.FACE_RECT_COLOR: Tuple[int, int, int] = (0, 255, 0)  # Green
        self.EYE_RECT_COLOR: Tuple[int, int, int] = (255, 0, 0)  # Blue

        # Default filter parameters
        self.DEFAULT_FILTER_PARAMS: Dict[str, Dict[str, Any]] = {
            "Gaussian Blur": {"kernel_size": 5, "sigma": 0},
            "Edge Detection": {
                "threshold1": 100,
                "threshold2": 200,
                "edge_mode": "Canny",
            },
            "Brightness & Contrast": {
                "brightness": 1.0,
                "contrast": 1.0,
                "saturation": 1.0,
            },
        }

        # Webcam settings
        self.DEFAULT_WEBCAM_MODE = "Continuous Detection"
        self.WEBCAM_FPS_TARGET = 15  # Target FPS for webcam processing
