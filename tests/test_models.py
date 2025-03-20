"""Tests for model handlers."""

import pytest
from unittest.mock import MagicMock, patch

from qllama.models import get_model_handler
from qllama.models.base import BaseModelHandler

def test_get_model_handler():
    """Test the get_model_handler function."""
    with pytest.raises(ValueError):
        get_model_handler("non_existent_model")
    
    with patch("qllama.models.import_module") as mock_import:
        mock_module = MagicMock()
        mock_handler = MagicMock(spec=BaseModelHandler)
        mock_module.SmolVLMHandler = mock_handler
        mock_import.return_value = mock_module
        
        handler = get_model_handler("smolvlm2")
        assert handler is mock_handler.return_value
        mock_handler.assert_called_once()

@pytest.mark.parametrize(
    "model_name", 
    ["smolvlm2", "SmolVLM2", "smol-vlm2", "SmolVLM2-2.2B"]
)
def test_model_name_normalization(model_name):
    """Test that different model name formats are normalized correctly."""
    with patch("qllama.models.import_module") as mock_import:
        mock_module = MagicMock()
        mock_handler = MagicMock(spec=BaseModelHandler)
        mock_module.SmolVLMHandler = mock_handler
        mock_import.return_value = mock_module
        
        get_model_handler(model_name)
        # The model name is normalized correctly if we get here without errors
        assert True
