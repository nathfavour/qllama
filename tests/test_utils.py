"""Tests for utility functions."""

import pytest
from unittest.mock import patch, MagicMock

from qllama.utils import is_url, load_image, load_video

def test_is_url():
    """Test is_url function."""
    assert is_url("http://example.com")
    assert is_url("https://example.com/image.jpg")
    assert not is_url("path/to/file.jpg")
    assert not is_url("/home/user/image.jpg")

@patch("qllama.utils.requests.get")
@patch("qllama.utils.Image")
@patch("qllama.utils.tempfile.NamedTemporaryFile")
def test_load_image_from_url(mock_tempfile, mock_pil, mock_requests):
    """Test loading an image from a URL."""
    # Setup mocks
    mock_response = MagicMock()
    mock_requests.return_value = mock_response
    
    mock_file = MagicMock()
    mock_file.name = "/tmp/temp_image.jpg"
    mock_tempfile.return_value.__enter__.return_value = mock_file
    
    mock_image = MagicMock()
    mock_pil.open.return_value = mock_image
    
    # Call the function
    result = load_image("https://example.com/image.jpg")
    
    # Assertions
    mock_requests.assert_called_once_with("https://example.com/image.jpg", stream=True)
    mock_response.raise_for_status.assert_called_once()
    mock_pil.open.assert_called_once_with("/tmp/temp_image.jpg")
    assert result == mock_image

@patch("qllama.utils.os.path.isfile")
@patch("qllama.utils.Image")
def test_load_image_from_path(mock_pil, mock_isfile):
    """Test loading an image from a path."""
    # Setup mocks
    mock_isfile.return_value = True
    mock_image = MagicMock()
    mock_pil.open.return_value = mock_image
    
    # Call the function
    result = load_image("/path/to/image.jpg")
    
    # Assertions
    mock_isfile.assert_called_once_with("/path/to/image.jpg")
    mock_pil.open.assert_called_once_with("/path/to/image.jpg")
    assert result == mock_image

@patch("qllama.utils.os.path.isfile")
@patch("qllama.utils.cv2")
@patch("qllama.utils.Image")
@patch("qllama.utils.np")
def test_load_video(mock_np, mock_pil, mock_cv2, mock_isfile):
    """Test loading video frames."""
    # Setup mocks
    mock_isfile.return_value = True
    
    mock_video = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_video
    mock_video.get.return_value = 10  # 10 frames
    
    mock_np.linspace.return_value = [0, 3, 6, 9]  # 4 frame indices
    
    frame_data = [MagicMock() for _ in range(4)]
    pil_images = [MagicMock() for _ in range(4)]
    
    mock_video.read.side_effect = [(True, frame) for frame in frame_data]
    mock_pil.fromarray.side_effect = pil_images
    
    # Call the function
    result = load_video("/path/to/video.mp4")
    
    # Assertions
    mock_isfile.assert_called_once_with("/path/to/video.mp4")
    mock_cv2.VideoCapture.assert_called_once_with("/path/to/video.mp4")
    assert mock_video.read.call_count == 4
    assert mock_cv2.cvtColor.call_count == 4
    assert mock_pil.fromarray.call_count == 4
    assert len(result) == 4
    assert result == pil_images
