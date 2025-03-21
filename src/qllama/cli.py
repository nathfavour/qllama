"""Command-line interface for qllama."""

import argparse
import logging
import sys
import os
from typing import List, Optional

from qllama import __version__
from qllama.terminal import QllamaTerminal

_logger = logging.getLogger(__name__)

def setup_logging(loglevel: int) -> None:
    """Setup basic logging.
    
    Args:
        loglevel: Minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel,
        stream=sys.stdout,
        format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse command line parameters.
    
    Args:
        args: Command line parameters as list of strings
        
    Returns:
        Command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="qllama - an alternative to ollama with low-level model access")
    parser.add_argument(
        "--version",
        action="version",
        version=f"qllama {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a model in interactive mode")
    run_parser.add_argument("model", help="Model name to run")
    run_parser.add_argument("--device", help="Device to use (cpu, cuda)", default="cuda")
    run_parser.add_argument("--temperature", type=float, help="Temperature for generation", default=1.0)
    run_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate", default=64)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    
    # Help command
    help_parser = subparsers.add_parser("help", help="Show help for a command")
    
    return parser.parse_args(args)

def main(args: List[str]) -> None:
    """Main entry point allowing external calls.
    
    Args:
        args: Command line parameters as list of strings
    """
    parsed_args = parse_args(args)
    setup_logging(parsed_args.loglevel)
    
    _logger.debug("Starting qllama...")
    
    if parsed_args.command == "run":
        try:
            term = QllamaTerminal(
                model_name=parsed_args.model,
                device=parsed_args.device,
                temperature=parsed_args.temperature,
                max_tokens=parsed_args.max_tokens,
            )
            term.run()
        except ImportError as e:
            if "PIL" in str(e) or "_imaging" in str(e):
                print("\nError: PIL (Pillow) is not properly installed.")
                print("This is most likely due to missing system dependencies.")
                print("\nTry fixing with this command:")
                print("    pip uninstall -y pillow && pip install --no-cache-dir pillow")
                print("\nOr run the fix script:")
                script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                          "scripts", "fix_pillow.sh")
                print(f"    bash {script_path}")
                print("\nIf that doesn't work, you may need to install system dependencies:")
                print("    Ubuntu/Debian: sudo apt-get install python3-dev libjpeg-dev libfreetype6-dev zlib1g-dev")
                print("    CentOS/RHEL: sudo yum install python3-devel libjpeg-devel freetype-devel zlib-devel")
                sys.exit(1)
            else:
                # Re-raise if it's not a PIL-related issue
                raise
    elif parsed_args.command == "list":
        try:
            from qllama.models import MODEL_REGISTRY
            print("Available models:")
            for model in MODEL_REGISTRY:
                print(f"  - {model}")
        except ImportError as e:
            if "PIL" in str(e) or "_imaging" in str(e):
                print("\nError: PIL (Pillow) is not properly installed.")
                print("Run 'pip uninstall -y pillow && pip install --no-cache-dir pillow' to fix the issue.")
                sys.exit(1)
            else:
                raise
    elif parsed_args.command == "help" or parsed_args.command is None:
        parse_args(["--help"])
    else:
        _logger.error(f"Unknown command: {parsed_args.command}")
        parse_args(["--help"])

def run() -> None:
    """Entry point for console_scripts."""
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
