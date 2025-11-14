#!/bin/bash

# NVIDIA Control Panel Uninstallation Script

set -e

INSTALL_DIR="/usr/local/bin"
SERVICE_FILE="/etc/systemd/system/nvidia-settings-persistence.service"
CONFIG_DIR="$HOME/.config/nvidiacp"

echo "======================================"
echo "NVIDIA Control Panel Uninstaller"
echo "======================================"
echo

# Check for sudo privileges
if [ "$EUID" -ne 0 ]; then 
    echo "This script requires sudo privileges for uninstallation."
    echo "Please run: sudo ./uninstall.sh"
    exit 1
fi

echo "This will uninstall the NVIDIA Control Panel."
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstallation cancelled."
    exit 0
fi

echo
echo "Uninstalling NVIDIA Control Panel..."
echo

# Stop and disable systemd service if it exists
if [ -f "$SERVICE_FILE" ]; then
    echo "Removing systemd service..."
    systemctl stop nvidia-settings-persistence.service 2>/dev/null || true
    systemctl disable nvidia-settings-persistence.service 2>/dev/null || true
    rm -f "$SERVICE_FILE"
    systemctl daemon-reload
    echo "✓ Removed systemd service"
fi

# Remove symlink
if [ -L "$INSTALL_DIR/nvidiacp" ]; then
    rm -f "$INSTALL_DIR/nvidiacp"
    echo "✓ Removed nvidiacp command"
fi

# Ask about config files
if [ -d "$CONFIG_DIR" ]; then
    echo
    read -p "Remove configuration files? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$CONFIG_DIR"
        echo "✓ Removed configuration files"
    else
        echo "Configuration files preserved at: $CONFIG_DIR"
    fi
fi

echo
echo "======================================"
echo "Uninstallation Complete!"
echo "======================================"
echo
echo "The NVIDIA Control Panel has been uninstalled."
echo "The application files in this directory have been preserved."
echo