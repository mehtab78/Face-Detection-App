"""Tests for configuration module."""
import os
import pytest
from src.config import Config


def test_config_initialization():
    """Test that Config initializes correctly."""
    config = Config()
    
    assert config.MIN_NEIGHBORS == 5
    assert config.SCALE_FACTOR == 1.1
    assert config.WEBCAM_FPS_TARGET == 15


def test_config_paths_exist():
    """Test that required paths are set."""
    config = Config()
    
    assert config.PROJECT_ROOT is not None
    assert config.HAAR_CASCADE_DIR is not None
    assert config.DATA_DIR is not None
    assert config.IMAGES_DIR is not None


def test_cascade_paths_exist():
    """Test that cascade paths are valid (either local or OpenCV built-in)."""
    config = Config()
    
    # These should exist (either local or fallback to OpenCV built-in)
    assert os.path.exists(config.FACE_CASCADE_PATH)
    assert os.path.exists(config.EYE_CASCADE_PATH)


def test_config_colors():
    """Test that colors are tuples of correct length."""
    config = Config()
    
    assert isinstance(config.FACE_RECT_COLOR, tuple)
    assert len(config.FACE_RECT_COLOR) == 3
    assert isinstance(config.EYE_RECT_COLOR, tuple)
    assert len(config.EYE_RECT_COLOR) == 3


def test_default_filter_params():
    """Test that default filter parameters are properly structured."""
    config = Config()
    
    assert "Gaussian Blur" in config.DEFAULT_FILTER_PARAMS
    assert "Edge Detection" in config.DEFAULT_FILTER_PARAMS
    assert "Brightness & Contrast" in config.DEFAULT_FILTER_PARAMS
    
    # Check Gaussian Blur params
    assert "kernel_size" in config.DEFAULT_FILTER_PARAMS["Gaussian Blur"]
    assert "sigma" in config.DEFAULT_FILTER_PARAMS["Gaussian Blur"]
