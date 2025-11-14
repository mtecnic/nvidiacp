# NVIDIA Control Panel - Installation & Usage Guide

## Quick Start (Recommended)

```bash
# 1. Clone or download the repository and navigate to it
cd nvidiacp

# 2. Run the installer (requires sudo)
sudo ./install.sh

# 3. Run the control panel
sudo nvidiacp
```

That's it! The program is now installed system-wide and ready to use.

---

## Manual Installation (Alternative)

If you prefer not to use the installer:

```bash
# 1. Navigate to the nvidiacp directory
cd nvidiacp

# 2. Make scripts executable
chmod +x nvidia_control.py
chmod +x vllm_optimizer.py

# 3. Run directly
sudo python3 ./nvidia_control.py
```

---

## How to Use

### Main Control Panel
```bash
sudo nvidiacp
# or (if not installed system-wide)
sudo python3 ./nvidia_control.py
```

### vLLM Optimizer (Standalone)
```bash
# Access via main menu (option 23) or run directly:
sudo python3 ./vllm_optimizer.py
```

### Apply Saved Settings (Used by boot service)
```bash
nvidiacp --apply-settings
```

---

## What the Installer Does

The `install.sh` script:
- ✅ Creates a system-wide `nvidiacp` command
- ✅ Installs systemd service for boot persistence  
- ✅ Enables automatic settings restoration on reboot
- ✅ Sets proper permissions
- ✅ Verifies NVIDIA drivers are installed

---

## Requirements

### Must Have:
- NVIDIA GPU with drivers installed
- `nvidia-smi` command working
- Python 3.6 or newer
- `sudo` access (for GPU settings)

### Check Requirements:
```bash
# Test NVIDIA drivers
nvidia-smi

# Test Python
python3 --version

# Test sudo access
sudo echo "Sudo works!"
```

---

## First Run

1. **Launch the program:**
   ```bash
   sudo nvidiacp
   ```

2. **You'll see a menu with 23 options organized by category:**
   - Information & Monitoring (1-5)
   - Power & Performance (6-11)
   - Memory & Compute (12-16)
   - Hardware Control (17-18)
   - System Management (19-22)
   - Specialized Tools (23)

3. **For vLLM users:** Select option 23 for vLLM-specific optimizations

4. **Settings automatically persist** - any changes you make will be restored on reboot

---

## Multi-GPU Setup

The program automatically detects multiple GPUs:

- **Single GPU:** Settings apply directly to GPU 0
- **Multiple GPUs:** You get a selection menu:
  ```
  Select GPU:
  0. GPU 0
  1. GPU 1
  2. All GPUs  ← Apply to all at once
  ```

Each GPU maintains separate settings that persist independently.

---

## Common Tasks

### Set Power Limits
1. Run `sudo nvidiacp`
2. Choose option 7 (Set Power Limit)
3. Select GPU or "All GPUs"
4. Enter wattage (e.g., 250)

### Optimize for vLLM
1. Run `sudo nvidiacp`
2. Choose option 23 (vLLM Optimization Tool)
3. Select optimization profile
4. Follow recommendations

### View Current Settings
1. Run `sudo nvidiacp`
2. Choose option 21 (Show Current Settings)

### Reset Everything
1. Run `sudo nvidiacp`
2. Choose option 20 (Reset All GPUs to Defaults)

---

## Boot Persistence

After installation, your GPU settings automatically restore on boot via systemd service:

### Check Service Status:
```bash
sudo systemctl status nvidia-settings-persistence.service
```

### Manually Apply Settings:
```bash
sudo systemctl start nvidia-settings-persistence.service
```

### Disable Auto-Restore:
```bash
sudo systemctl disable nvidia-settings-persistence.service
```

### Re-Enable Auto-Restore:
```bash
sudo systemctl enable nvidia-settings-persistence.service
```

---

## Uninstallation

```bash
# Navigate to the nvidiacp directory
cd nvidiacp
sudo ./uninstall.sh
```

This removes:
- The `nvidiacp` command
- The systemd service
- Optionally removes your saved settings

---

## Troubleshooting

### "Permission Denied"
Most NVIDIA settings require root privileges:
```bash
sudo nvidiacp  # Always use sudo
```

### "nvidia-smi not found"
Install NVIDIA drivers:
```bash
# Check if drivers are installed
lspci | grep -i nvidia

# If you see NVIDIA cards but no nvidia-smi, install drivers
sudo apt update && sudo apt install nvidia-driver-535
```

### Settings Not Persisting
Check if the systemd service is running:
```bash
sudo systemctl status nvidia-settings-persistence.service
```

### vLLM Optimizer Not Found
Make sure vllm_optimizer.py is in the same directory as nvidia_control.py:
```bash
ls -la vllm_optimizer.py
```

### Fan Control Not Working
Fan control requires X11 and coolbits configuration. This is optional and mainly for desktop use.

---

## File Locations

### Program Files:
- `nvidia_control.py` - Main application
- `vllm_optimizer.py` - vLLM optimizer
- `install.sh` - Installer script
- `uninstall.sh` - Uninstaller script

### Configuration Files:
- `~/.config/nvidiacp/settings.json` - Your saved GPU settings
- `~/.config/nvidiacp/vllm_settings.json` - vLLM optimizer settings

### System Files (Created by installer):
- `/usr/local/bin/nvidiacp` - System-wide command (symlink to nvidia_control.py)
- `/etc/systemd/system/nvidia-settings-persistence.service` - Boot persistence service

---

## Safety Notes

- ⚠️ **Always use sudo** - GPU settings require root access
- ⚠️ **ECC changes require reboot** - Plan accordingly  
- ⚠️ **Power limits are enforced** - Don't exceed your PSU capacity
- ⚠️ **Reset option available** - Option 20 resets everything to defaults if needed

---

## Support

If something isn't working:

1. **Check nvidia-smi works:** `nvidia-smi`
2. **Check GPU detection:** `nvidia-smi -L`
3. **Run with verbose output:** Add debug prints if needed
4. **Reset to defaults:** Use option 20 in the menu

The program is designed to be safe - it won't let you set invalid values and provides clear error messages when something fails.