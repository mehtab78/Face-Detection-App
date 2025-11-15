"""Tests for visualization utilities."""
import numpy as np
import pytest
import tempfile
import os
from src.utils.visualization import (
    draw_detections,
    draw_facial_features,
    save_image,
)


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    # Create a 300x300 BGR image
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[:, :] = (100, 100, 100)  # Gray
    return image


@pytest.fixture
def sample_faces():
    """Create sample face detections."""
    # Array of face rectangles (x, y, w, h)
    faces = np.array([[50, 50, 100, 100], [200, 200, 80, 80]])
    return faces


def test_draw_detections_shape(sample_image, sample_faces):
    """Test that draw_detections preserves image shape."""
    result = draw_detections(sample_image, sample_faces)
    
    assert result.shape == sample_image.shape


def test_draw_detections_modifies_image(sample_image, sample_faces):
    """Test that draw_detections actually draws rectangles."""
    result = draw_detections(sample_image, sample_faces)
    
    # Result should differ from original
    assert not np.array_equal(result, sample_image)


def test_draw_detections_empty_faces(sample_image):
    """Test draw_detections with no faces."""
    empty_faces = np.array([]).reshape(0, 4)
    result = draw_detections(sample_image, empty_faces)
    
    # Should return image unchanged
    assert np.array_equal(result, sample_image)


def test_draw_facial_features_structure(sample_image, sample_faces):
    """Test draw_facial_features with proper structure."""
    detected_features = {
        "faces": sample_faces,
        "eyes": [[(60, 60, 20, 20), (100, 60, 20, 20)], []],
    }
    
    result = draw_facial_features(sample_image, detected_features)
    
    assert result.shape == sample_image.shape
    assert not np.array_equal(result, sample_image)


def test_save_image(sample_image):
    """Test saving image to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_image.png")
        
        # Save image
        success = save_image(sample_image, filepath)
        
        assert success is True
        assert os.path.exists(filepath)


def test_save_image_invalid_path(sample_image):
    """Test saving image to invalid path."""
    invalid_path = "/invalid/nonexistent/directory/image.png"
    
    success = save_image(sample_image, invalid_path)
    
    assert success is False
