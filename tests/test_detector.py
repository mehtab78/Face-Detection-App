"""Tests for face detector module."""
import numpy as np
import pytest
import cv2
from src.detector import FaceDetector
from src.config import Config


@pytest.fixture
def detector():
    """Create a FaceDetector instance for testing."""
    config = Config()
    return FaceDetector(config)


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    # Create a 300x300 BGR image (black)
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    return image


def test_detector_initialization(detector):
    """Test that detector initializes correctly."""
    assert detector is not None
    assert detector.face_cascade is not None
    assert detector.eye_cascade is not None
    assert detector.scale_factor == 1.1
    assert detector.min_neighbors == 5


def test_detector_with_config():
    """Test detector initialization with custom config."""
    config = Config()
    detector = FaceDetector(config)
    
    assert detector.config == config


def test_set_params(detector):
    """Test parameter setting."""
    detector.set_params(scale_factor=1.2, min_neighbors=7)
    
    assert detector.scale_factor == 1.2
    assert detector.min_neighbors == 7


def test_detect_faces_no_faces(detector, sample_image):
    """Test detection on image with no faces."""
    faces = detector.detect_faces(sample_image)
    
    assert isinstance(faces, np.ndarray)
    # Should detect no faces in blank image
    assert len(faces) == 0


def test_detect_faces_returns_correct_shape(detector, sample_image):
    """Test that detect_faces returns correct array shape."""
    faces = detector.detect_faces(sample_image)
    
    # Should be 2D array with 4 columns (x, y, w, h)
    if len(faces) > 0:
        assert faces.shape[1] == 4


def test_detect_facial_features_structure(detector, sample_image):
    """Test that detect_facial_features returns correct structure."""
    features = detector.detect_facial_features(sample_image)
    
    assert isinstance(features, dict)
    assert "faces" in features
    assert "eyes" in features
    assert isinstance(features["faces"], np.ndarray)
    assert isinstance(features["eyes"], list)


def test_property_getters(detector):
    """Test property getters."""
    assert detector.scale_factor == 1.1
    assert detector.min_neighbors == 5


def test_property_setters(detector):
    """Test property setters."""
    detector.scale_factor = 1.3
    detector.min_neighbors = 8
    
    assert detector.scale_factor == 1.3
    assert detector.min_neighbors == 8


def test_cascade_loading_error():
    """Test that invalid cascade path raises error."""
    config = Config()
    # Set invalid path
    config.FACE_CASCADE_PATH = "/invalid/path/to/cascade.xml"
    
    with pytest.raises(ValueError):
        FaceDetector(config)
