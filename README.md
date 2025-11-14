# NVIDIA Control Panel

A comprehensive text-based menu system for controlling NVIDIA GPU settings via nvidia-smi with automatic persistence on reboot.

## Features

- **Complete nvidia-smi Control**: Access all major nvidia-smi commands through an intuitive text menu
- **Settings Persistence**: Automatically save and restore GPU settings on system reboot
- **Multi-GPU Support**: Control individual GPUs or apply settings to all GPUs simultaneously
- **Systemd Integration**: Built-in systemd service for boot-time settings restoration

## Supported Settings

- **Persistence Mode**: Enable/disable driver persistence mode
- **Power Management**: Set power limits (watts)
- **Clock Speeds**: Configure memory and graphics clock speeds
- **Compute Mode**: Set compute mode (Default/Exclusive Thread/Prohibited/Exclusive Process)
- **Fan Control**: Adjust fan speeds (requires X11 and coolbits)
- **ECC Memory**: Enable/disable ECC (requires reboot)
- **Auto Boost**: Control GPU auto boost feature
- **Reset Functions**: Reset individual or all GPU settings to defaults

## Installation

### Quick Install (Recommended)

```bash
cd nvidiacp
sudo chmod +x install.sh
sudo ./install.sh
```

This will:
- Create a system-wide `nvidiacp` command
- Install and enable the systemd service for boot persistence
- Set up all necessary permissions

### Manual Installation

1. Make the script executable:
```bash
chmod +x nvidia_control.py
```

2. Run directly:
```bash
sudo python3 nvidia_control.py
```

3. For boot persistence, manually install the systemd service:
```bash
sudo cp nvidia-settings-persistence.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nvidia-settings-persistence.service
```

## Usage

### Interactive Mode

Run the control panel:
```bash
nvidiacp
# or
sudo python3 nvidia_control.py
```

Navigate through the menu using number keys to:
1. View current GPU information
2. Modify GPU settings
3. Save settings for persistence
4. Apply saved settings
5. Reset GPUs to defaults

### Command Line Mode

Apply saved settings (used by systemd service):
```bash
python3 nvidia_control.py --apply-settings
```

## Settings Storage

Settings are stored in JSON format at:
```
~/.config/nvidiacp/settings.json
```

Example settings file:
```json
{
  "gpu_0_persistence": true,
  "gpu_0_power_limit": 250,
  "gpu_0_mem_clock": 5001,
  "gpu_0_graphics_clock": 1500,
  "gpu_0_compute_mode": 0
}
```

## Systemd Service

The systemd service automatically applies your saved settings on boot.

### Service Management

```bash
# Check service status
sudo systemctl status nvidia-settings-persistence.service

# Manually trigger settings application
sudo systemctl start nvidia-settings-persistence.service

# Disable automatic application
sudo systemctl disable nvidia-settings-persistence.service

# Re-enable automatic application
sudo systemctl enable nvidia-settings-persistence.service
```

### Service Logs

View service logs:
```bash
journalctl -u nvidia-settings-persistence.service
```

## Uninstallation

```bash
cd nvidiacp
sudo chmod +x uninstall.sh
sudo ./uninstall.sh
```

This will:
- Remove the systemd service
- Remove the `nvidiacp` command
- Optionally remove configuration files

## Requirements

- NVIDIA GPU with NVIDIA drivers installed
- `nvidia-smi` command available
- Python 3.6+
- `sudo` privileges for system settings
- Optional: X11 with coolbits for fan control

## Troubleshooting

### Permission Denied
Most nvidia-smi commands require root privileges. Run with `sudo`:
```bash
sudo nvidiacp
```

### Fan Control Not Working
Fan control requires:
1. X11 display server running
2. Coolbits enabled in X11 configuration
3. nvidia-settings package installed

### Settings Not Persisting
1. Check systemd service status:
```bash
sudo systemctl status nvidia-settings-persistence.service
```

2. Verify settings file exists:
```bash
ls -la ~/.config/nvidiacp/settings.json
```

3. Check service logs for errors:
```bash
journalctl -u nvidia-settings-persistence.service -n 50
```

### ECC Memory Changes
ECC memory changes require a system reboot to take effect.

## Security Notes

- The application requires sudo privileges to modify GPU settings
- Settings are stored in user's home directory with standard permissions
- Systemd service runs as root to apply settings at boot

## License

This tool is provided as-is for system administration purposes.

## Support

For issues or questions:
1. Check nvidia-smi documentation: `man nvidia-smi`
2. Verify NVIDIA driver installation: `nvidia-smi`
3. Check system logs: `dmesg | grep -i nvidia`