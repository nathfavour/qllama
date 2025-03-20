#!/bin/bash
# Fix the PIL installation issue by reinstalling pillow

echo "Fixing PIL installation issue..."

# First, uninstall any existing pillow installations
pip uninstall -y pillow

# Install required system dependencies (if running as root or with sudo)
if [ "$(id -u)" = "0" ]; then
    echo "Installing system dependencies..."
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        apt-get update
        apt-get install -y python3-dev libjpeg-dev libfreetype6-dev zlib1g-dev
    elif [ -f /etc/redhat-release ]; then
        # CentOS/RHEL/Fedora
        yum install -y python3-devel libjpeg-devel freetype-devel zlib-devel
    fi
else
    echo "Note: Running as non-root user. If installation fails, you may need to install system dependencies."
    echo "For Ubuntu/Debian: sudo apt-get install python3-dev libjpeg-dev libfreetype6-dev zlib1g-dev"
    echo "For CentOS/RHEL: sudo yum install python3-devel libjpeg-devel freetype-devel zlib-devel"
fi

# Reinstall pillow with no cache
pip install --no-cache-dir pillow

# Verify installation
echo "Verifying PIL installation..."
python -c "from PIL import Image; print('PIL installation successful!')"

# If the package is installed successfully, exit with success
if [ $? -eq 0 ]; then
    echo "PIL has been successfully reinstalled."
    exit 0
else
    echo "PIL installation verification failed. You may need to install system dependencies first."
    exit 1
fi
