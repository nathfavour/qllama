"""Dependency checker for qllama."""

import logging
import importlib.util
import sys
from typing import Dict, List, Tuple

_logger = logging.getLogger(__name__)

def check_dependencies() -> List[Tuple[str, str, bool]]:
    """Check if all required dependencies are installed.
    
    Returns:
        List of (package_name, error_message, is_critical) tuples
    """
    issues = []
    
    # Check PIL/Pillow
    try:
        from PIL import Image
        # Try to verify _imaging is available
        img = Image.new("RGB", (1, 1))
        del img
    except ImportError as e:
        if "_imaging" in str(e):
            issues.append(("pillow", 
                          "PIL is not properly installed. This usually happens due to missing system dependencies. "
                          "Try reinstalling: pip uninstall -y pillow && pip install --no-cache-dir pillow", 
                          True))
        else:
            issues.append(("pillow", f"PIL import error: {e}", True))
    except Exception as e:
        issues.append(("pillow", f"PIL error: {e}", True))
    
    # Check torch
    try:
        import torch
    except ImportError as e:
        issues.append(("torch", f"PyTorch not installed: {e}", True))
    except Exception as e:
        issues.append(("torch", f"PyTorch error: {e}", True))
    
    # Check transformers
    try:
        import transformers
    except ImportError as e:
        issues.append(("transformers", f"Transformers not installed: {e}", True))
    except Exception as e:
        issues.append(("transformers", f"Transformers error: {e}", True))
    
    # Check optional dependencies
    # OpenCV
    try:
        import cv2
    except ImportError as e:
        issues.append(("opencv-python", f"OpenCV not installed: {e}", False))
    except Exception as e:
        issues.append(("opencv-python", f"OpenCV error: {e}", False))
    
    return issues

def check_and_report() -> bool:
    """Check dependencies and report issues.
    
    Returns:
        True if all critical dependencies are installed, False otherwise
    """
    issues = check_dependencies()
    
    if not issues:
        return True
    
    critical_issues = [issue for issue in issues if issue[2]]
    
    if critical_issues:
        _logger.error("Critical dependency issues found:")
        for package, msg, _ in critical_issues:
            _logger.error(f"  - {package}: {msg}")
        
        print("\nCritical dependency issues were detected:")
        for package, msg, _ in critical_issues:
            print(f"  - {package}: {msg}")
        
        print("\nPlease fix these issues before running qllama.")
        print("For PIL/Pillow issues, try: pip uninstall -y pillow && pip install --no-cache-dir pillow")
        return False
    
    # Log non-critical issues as warnings
    non_critical = [issue for issue in issues if not issue[2]]
    for package, msg, _ in non_critical:
        _logger.warning(f"{package}: {msg}")
    
    return True

if __name__ == "__main__":
    # Can be run directly to check dependencies
    logging.basicConfig(level=logging.INFO)
    if check_and_report():
        print("All critical dependencies are installed!")
    else:
        sys.exit(1)
