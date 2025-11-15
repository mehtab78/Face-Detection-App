"""Visualization utilities for drawing detection results."""
import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, List
from numpy.typing import NDArray

from ..config import Config

logger = logging.getLogger(__name__)

# Create a config instance to access the constants
config = Config()
FACE_RECT_COLOR: Tuple[int, int, int] = config.FACE_RECT_COLOR
EYE_RECT_COLOR: Tuple[int, int, int] = config.EYE_RECT_COLOR


def draw_detections(
    image: NDArray[np.uint8], faces: NDArray[np.int32]
) -> NDArray[np.uint8]:
    """Draw rectangles around detected faces.
    
    Args:
        image: Input image in BGR format
        faces: Array of face rectangles (x, y, w, h)
        
    Returns:
        Image with drawn rectangles
    """
    output = image.copy()
    for x, y, w, h in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), FACE_RECT_COLOR, 2)
    logger.debug(f"Drew {len(faces)} face rectangles")
    return output


def draw_facial_features(
    image: NDArray[np.uint8], detected_features: Dict[str, Any]
) -> NDArray[np.uint8]:
    """Draw rectangles around detected faces and eyes.
    
    Args:
        image: Input image in BGR format
        detected_features: Dictionary with 'faces' and 'eyes' keys
        
    Returns:
        Image with drawn features
    """
    # Create a copy of the image to draw on
    output = image.copy()

    # Draw rectangles around faces
    output = draw_detections(output, detected_features["faces"])

    # Draw rectangles around eyes
    for i, face in enumerate(detected_features["faces"]):
        if i < len(detected_features["eyes"]):
            for x, y, w, h in detected_features["eyes"][i]:
                cv2.rectangle(output, (x, y), (x + w, y + h), EYE_RECT_COLOR, 2)

    return output


def display_image(
    image: NDArray[np.uint8], window_name: str = "Facial Features Detection"
) -> None:
    """Display image in OpenCV window (for standalone scripts).
    
    Args:
        image: Image to display
        window_name: Window title
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(image: NDArray[np.uint8], output_path: str) -> bool:
    """Save image to disk.
    
    Args:
        image: Image to save
        output_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cv2.imwrite(output_path, image)
        logger.info(f"Image saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return False
