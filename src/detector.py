"""Face detection module using OpenCV Haar Cascades."""
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any
from numpy.typing import NDArray

from .config import Config

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Class for detecting facial features using Haar Cascades.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize face detector with Haar cascade classifiers.
        
        Args:
            config: Configuration object. If None, creates a new Config instance.
            
        Raises:
            ValueError: If cascade files cannot be loaded.
        """
        # Use provided config or create a new one
        self.config = config if config else Config()

        # Load the Haar cascade XML files
        self.face_cascade = cv2.CascadeClassifier(self.config.FACE_CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(self.config.EYE_CASCADE_PATH)

        # Check if the cascade files were loaded successfully
        if self.face_cascade.empty():
            error_msg = f"Error loading face cascade classifier from {self.config.FACE_CASCADE_PATH}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if self.eye_cascade.empty():
            error_msg = f"Error loading eye cascade classifier from {self.config.EYE_CASCADE_PATH}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Face detector initialized successfully")

        # Default detection parameters
        self._scale_factor = self.config.SCALE_FACTOR
        self._min_neighbors = self.config.MIN_NEIGHBORS

    def detect_faces(self, image: NDArray[np.uint8]) -> NDArray[np.int32]:
        """
        Detect faces in the input image.

        Args:
            image: Input image in BGR format (numpy array)

        Returns:
            Array of face rectangles with shape (n, 4) where each row is [x, y, w, h]
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(30, 30),
        )

        logger.debug(f"Detected {len(faces)} faces")
        return faces

    def detect_eyes(
        self, image: NDArray[np.uint8], face_roi: NDArray[np.uint8]
    ) -> NDArray[np.int32]:
        """
        Detect eyes within a face region.

        Args:
            image: Input image in BGR format
            face_roi: Face region of interest

        Returns:
            Array of eye rectangles with shape (n, 4) where each row is [x, y, w, h]
        """
        # Convert to grayscale
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(20, 20),
        )

        return eyes

    def detect_facial_features(
        self, image: NDArray[np.uint8]
    ) -> Dict[str, Any]:
        """
        Detect faces and eyes in an image.

        Args:
            image: Input image in BGR format

        Returns:
            Dictionary containing:
                - 'faces': Array of face rectangles
                - 'eyes': List of eye rectangle arrays for each face
        """
        # Make a copy of the image to avoid modifying the original
        img_copy = image.copy()

        # Detect faces first
        faces = self.detect_faces(img_copy)

        # Dictionary to store all detected features
        detected_features: Dict[str, Any] = {"faces": faces, "eyes": []}

        # For each detected face, detect eyes
        for x, y, w, h in faces:
            # Extract face region
            face_roi = img_copy[y : y + h, x : x + w]

            # Detect eyes in the face region
            eyes = self.detect_eyes(img_copy, face_roi)

            # Add eye coordinates relative to the whole image
            detected_features["eyes"].append(
                [(ex + x, ey + y, ew, eh) for (ex, ey, ew, eh) in eyes]
            )

        logger.info(f"Detected {len(faces)} faces with {sum(len(e) for e in detected_features['eyes'])} eyes")
        return detected_features

    @property
    def scale_factor(self) -> float:
        """Get current scale factor."""
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value: float) -> None:
        """Set scale factor."""
        self._scale_factor = value

    @property
    def min_neighbors(self) -> int:
        """Get current min neighbors value."""
        return self._min_neighbors

    @min_neighbors.setter
    def min_neighbors(self, value: int) -> None:
        """Set min neighbors value."""
        self._min_neighbors = value

    def set_params(
        self, scale_factor: Optional[float] = None, min_neighbors: Optional[int] = None
    ) -> None:
        """Update detector parameters.
        
        Args:
            scale_factor: Detection scale factor (1.05-1.5)
            min_neighbors: Minimum neighbors parameter (1-10)
        """
        if scale_factor is not None:
            self._scale_factor = scale_factor
        if min_neighbors is not None:
            self._min_neighbors = min_neighbors
        logger.debug(f"Updated params: scale_factor={self._scale_factor}, min_neighbors={self._min_neighbors}")
