#!/bin/bash

# NVIDIA Control Panel Installation Script

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR="/usr/local/bin"
SERVICE_FILE="/etc/systemd/system/nvidia-settings-persistence.service"

echo "======================================"
echo "NVIDIA Control Panel Installer"
echo "======================================"
echo

# Check for NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check for sudo privileges
if [ "$EUID" -ne 0 ]; then 
    echo "This script requires sudo privileges for installation."
    echo "Please run: sudo ./install.sh"
    exit 1
fi

echo "Installing NVIDIA Control Panel..."
echo

# Make the main script executable
chmod +x "$SCRIPT_DIR/nvidia_control.py"

# Create symlink in /usr/local/bin for easy access
ln -sf "$SCRIPT_DIR/nvidia_control.py" "$INSTALL_DIR/nvidiacp"
echo "✓ Created symlink: nvidiacp -> $SCRIPT_DIR/nvidia_control.py"

# Create the shared system settings directory (read/written by both the
# interactive app and the root boot service).
mkdir -p /etc/nvidiacp
echo "✓ Created settings directory: /etc/nvidiacp"

# Install systemd service for persistence
echo
echo "Installing systemd service for boot persistence..."

# Hardened unit: runs as root, waits for the driver / persistence daemon, and
# has an explicit PATH so nvidia-smi is found in the boot environment. Keep this
# in sync with generate_systemd_service() in nvidia_control.py.
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=NVIDIA GPU Settings Persistence
After=multi-user.target nvidia-persistenced.service
Wants=nvidia-persistenced.service
ConditionPathExists=/usr/bin/nvidia-smi

[Service]
Type=oneshot
User=root
Environment="PATH=/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
ExecStartPre=/bin/sleep 5
ExecStart=/usr/bin/python3 $SCRIPT_DIR/nvidia_control.py --apply-settings
RemainAfterExit=yes
StandardOutput=journal
StandardError=journal
TimeoutStartSec=60

[Install]
WantedBy=multi-user.target
EOF

echo "✓ Created systemd service: $SERVICE_FILE"

# Reload systemd daemon
systemctl daemon-reload
echo "✓ Reloaded systemd daemon"

# Enable the service (but don't start it yet)
systemctl enable nvidia-settings-persistence.service
echo "✓ Enabled nvidia-settings-persistence service"

echo
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo
echo "Usage:"
echo "  - Run the control panel: nvidiacp"
echo "  - Or directly: python3 $SCRIPT_DIR/nvidia_control.py"
echo
echo "The systemd service has been installed and enabled."
echo "Your GPU settings will be automatically applied on boot."
echo
echo "To manually start the persistence service:"
echo "  sudo systemctl start nvidia-settings-persistence.service"
echo
echo "To check service status:"
echo "  sudo systemctl status nvidia-settings-persistence.service"
echo