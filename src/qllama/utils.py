"""Utilities for qllama."""

import os
import logging
import tempfile
from typing import List, Union, Optional, Any
from urllib.parse import urlparse
import requests
from pathlib import Path

_logger = logging.getLogger(__name__)

# Verify PIL is properly installed at module import time
try:
    from PIL import Image
except ImportError as e:
    if "_imaging" in str(e):
        _logger.error("PIL is not properly installed. This is usually caused by missing dependencies.")
        _logger.error("Try reinstalling pillow with: pip uninstall -y pillow && pip install --no-cache-dir pillow")
    raise

def is_url(path_or_url: str) -> bool:
    """Check if a string is a URL.
    
    Args:
        path_or_url: A string to check
        
    Returns:
        True if the string is a URL, False otherwise
    """
    try:
        result = urlparse(path_or_url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def load_image(path_or_url: str) -> Any:
    """Load an image from a path or URL.
    
    Args:
        path_or_url: Path to an image file or URL
        
    Returns:
        A loaded image object (PIL Image or as required by the model)
    """
    try:
        if is_url(path_or_url):
            _logger.debug(f"Loading image from URL: {path_or_url}")
            response = requests.get(path_or_url, stream=True)
            response.raise_for_status()
            
            # Create a temporary file to save the downloaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                tmp_path = tmp.name
            
            # Load the image using PIL
            image = Image.open(tmp_path)
            
            # Clean up the temporary file
            os.unlink(tmp_path)
            
            return image
        else:
            _logger.debug(f"Loading image from path: {path_or_url}")
            # Check if the file exists
            if not os.path.isfile(path_or_url):
                raise FileNotFoundError(f"Image file not found: {path_or_url}")
            
            # Load the image using PIL
            return Image.open(path_or_url)
        
    except Exception as e:
        _logger.error(f"Error loading image from {path_or_url}: {e}")
        raise

def load_video(path: str, max_frames: int = 8) -> List[Any]:
    """Load video frames from a path.
    
    Args:
        path: Path to a video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        A list of frames as PIL Images
    """
    try:
        import cv2
        from PIL import Image
        import numpy as np
        
        _logger.debug(f"Loading video from path: {path}")
        
        # Check if the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        
        # Open the video file
        video = cv2.VideoCapture(path)
        
        frames = []
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            raise ValueError(f"No frames found in video: {path}")
        
        # Calculate frame indices to extract (evenly distributed)
        indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        for i in indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
        
        video.release()
        
        if not frames:
            raise ValueError(f"Failed to extract frames from video: {path}")
        
        return frames
        
    except Exception as e:
        _logger.error(f"Error loading video from {path}: {e}")
        raise
