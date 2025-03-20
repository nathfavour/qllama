"""Tests for command-line interface."""

import pytest
from unittest.mock import patch

from qllama.cli import parse_args, main

def test_parse_args():
    """Test argument parsing."""
    args = parse_args(["run", "smolvlm2"])
    assert args.command == "run"
    assert args.model == "smolvlm2"
    
    args = parse_args(["list"])
    assert args.command == "list"

@patch("qllama.cli.QllamaTerminal")
def test_main_run(mock_terminal):
    """Test main function with run command."""
    mock_terminal_instance = mock_terminal.return_value
    
    main(["run", "smolvlm2"])
    
    mock_terminal.assert_called_once_with(
        model_name="smolvlm2",
        device="cuda",
        temperature=1.0,
        max_tokens=64,
    )
    mock_terminal_instance.run.assert_called_once()

@patch("builtins.print")
def test_main_list(mock_print):
    """Test main function with list command."""
    with patch("qllama.cli.MODEL_REGISTRY", {"model1": "path1", "model2": "path2"}):
        main(["list"])
    
    # Check that print was called with expected outputs
    assert mock_print.call_args_list[0][0][0] == "Available models:"
    assert any("model1" in call[0][0] for call in mock_print.call_args_list)
    assert any("model2" in call[0][0] for call in mock_print.call_args_list)
