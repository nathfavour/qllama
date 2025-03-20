#!/bin/bash
# Update dependencies for qllama

# Update pip and setuptools
pip install -U pip setuptools wheel

# Install development dependencies
pip install -U pytest pytest-cov flake8 black

# Install runtime dependencies
pip install -U torch>=2.0.0 transformers>=4.30.0 pillow>=9.0.0 requests>=2.28.0 opencv-python>=4.5.0 tqdm>=4.64.0

# Install package in development mode
pip install -e .

echo "Dependencies updated successfully!"
