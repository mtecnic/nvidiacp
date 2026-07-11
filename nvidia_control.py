#!/usr/bin/env python3
"""
NVIDIA GPU Control Panel
A text-based menu system for controlling nvidia-smi settings with persistence
"""

import subprocess
import json
import os
import sys
import time
import glob
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Hardcoded fallback specs, keyed by a substring of the GPU name (nvidia-smi).
# Only used when a live driver query returns nothing usable (e.g. empty
# SUPPORTED_CLOCKS or a missing power range). Values are approximate maxima.
GPU_SPECS = {
    'H100':      {'core_max': 1980, 'mem_max': 2619, 'power_max': 700},
    'A100':      {'core_max': 1410, 'mem_max': 1593, 'power_max': 400},
    'RTX 6000 Ada': {'core_max': 2505, 'mem_max': 10001, 'power_max': 300},
    'RTX A6000': {'core_max': 1800, 'mem_max': 8001,  'power_max': 300},
    'A40':       {'core_max': 1740, 'mem_max': 7251,  'power_max': 300},
    'RTX A5000': {'core_max': 1695, 'mem_max': 8001,  'power_max': 230},
    'RTX A4000': {'core_max': 1560, 'mem_max': 7001,  'power_max': 140},
    'RTX 4090':  {'core_max': 2520, 'mem_max': 10501, 'power_max': 450},
    'RTX 4080':  {'core_max': 2505, 'mem_max': 11201, 'power_max': 320},
    'RTX 3090':  {'core_max': 1965, 'mem_max': 9751,  'power_max': 350},
    'RTX 3080':  {'core_max': 1935, 'mem_max': 9501,  'power_max': 320},
}


# vLLM tuning profiles. Honest about what actually applies on modern
# (Ampere/Ada/Hopper) hardware: no legacy auto-boost, ECC is opt-in only,
# compute mode stays Default (what vLLM wants), and efficiency profiles cap
# power rather than pinning clocks.
VLLM_PROFILES = {
    'max_throughput': {
        'name': 'Maximum Throughput',
        'description': 'Highest aggregate tokens/s — lock max clocks, full power',
        'power_pct': 100, 'lock_max_clocks': True, 'mem_util': 0.95,
        'extra_flags': ['--enable-chunked-prefill', '--enable-prefix-caching'],
    },
    'low_latency': {
        'name': 'Low Latency',
        'description': 'Minimal first-token latency — max clocks, eager execution',
        'power_pct': 100, 'lock_max_clocks': True, 'mem_util': 0.90,
        'extra_flags': ['--enforce-eager'],
    },
    'balanced': {
        'name': 'Balanced',
        'description': 'Good throughput with thermal headroom (default clocks)',
        'power_pct': 90, 'lock_max_clocks': False, 'mem_util': 0.90,
        'extra_flags': [],
    },
    'power_efficient': {
        'name': 'Power Efficient',
        'description': 'Cap power (not clocks) for perf/watt — let boost manage',
        'power_pct': 70, 'lock_max_clocks': False, 'mem_util': 0.85,
        'extra_flags': [],
    },
    'production': {
        'name': 'Production Ready',
        'description': 'Stable 90% power, default clocks, persistence on',
        'power_pct': 90, 'lock_max_clocks': False, 'mem_util': 0.90,
        'extra_flags': [],
    },
}


def match_gpu_spec(name: str) -> Optional[Dict[str, int]]:
    """Return the hardcoded fallback spec whose key is a substring of name."""
    if not name:
        return None
    # Prefer the longest matching key so 'RTX 6000 Ada' beats a looser match.
    for key in sorted(GPU_SPECS, key=len, reverse=True):
        if key in name:
            return GPU_SPECS[key]
    return None


def parse_supported_clocks(output: str) -> Tuple[List[int], Dict[int, List[int]]]:
    """Parse `nvidia-smi -q -d SUPPORTED_CLOCKS` output.

    Returns (mem_clocks, graphics_clocks) where mem_clocks is the ordered list
    of supported memory clocks (MHz) and graphics_clocks maps each memory clock
    to its list of supported graphics clocks (MHz).
    """
    mem_clocks: List[int] = []
    graphics_clocks: Dict[int, List[int]] = {}
    current_mem: Optional[int] = None

    for line in output.split('\n'):
        line = line.strip()
        if 'Memory' in line and 'MHz' in line:
            try:
                current_mem = int(line.split()[2])
                mem_clocks.append(current_mem)
                graphics_clocks[current_mem] = []
            except (ValueError, IndexError):
                pass
        elif 'Graphics' in line and 'MHz' in line and current_mem is not None:
            try:
                graphics_clocks[current_mem].append(int(line.split()[2]))
            except (ValueError, IndexError):
                pass
    return mem_clocks, graphics_clocks


class Style:
    """Tiny ANSI styling helper. Colors auto-disable when output is not a
    terminal or when NO_COLOR is set."""
    enabled = sys.stdout.isatty() and os.environ.get('NO_COLOR') is None

    CODES = {
        'reset': '0', 'bold': '1', 'dim': '2',
        'red': '31', 'green': '32', 'yellow': '33', 'blue': '34',
        'magenta': '35', 'cyan': '36', 'white': '37', 'gray': '90',
        'bgblue': '44',
    }

    @classmethod
    def paint(cls, text: str, *styles: str) -> str:
        if not cls.enabled or not styles:
            return text
        codes = ';'.join(cls.CODES[s] for s in styles if s in cls.CODES)
        return f"\033[{codes}m{text}\033[0m"

    @classmethod
    def strip_len(cls, text: str) -> int:
        """Visible length of a string, ignoring ANSI escape codes."""
        return len(re.sub(r'\033\[[0-9;]*m', '', text))


def c(text: str, *styles: str) -> str:
    return Style.paint(text, *styles)


class NvidiaGPUController:
    def __init__(self):
        self.settings_file = Path.home() / '.config' / 'nvidiacp' / 'settings.json'
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings = self.load_settings()
        self._migrate_settings()
        self.gpus = self.detect_gpus()
        self.gpu_count = len(self.gpus)
        self.driver_version, self.cuda_version = self.get_driver_info()

    def load_settings(self) -> Dict[str, Any]:
        """Load persistent settings from file"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}
    
    def _migrate_settings(self):
        """Migrate legacy application-clock keys (gpu_N_app_*_clock, applied via
        the deprecated -ac API that no-ops on GeForce) to the new locked-clock
        keys (gpu_N_lock_*_clock, applied via -lgc/-lmc). Preserves the user's
        intent so a saved core-clock actually takes effect on the fixed code."""
        rename = {}
        for key in list(self.settings):
            if key.endswith('_app_graphics_clock'):
                rename[key] = key.replace('_app_graphics_clock', '_lock_gpu_clock')
            elif key.endswith('_app_mem_clock'):
                rename[key] = key.replace('_app_mem_clock', '_lock_mem_clock')
        if rename:
            for old, new in rename.items():
                self.settings[new] = self.settings.pop(old)
            self.save_settings()

    def save_settings(self):
        """Save current settings to file"""
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=2)
    
    def run_command(self, cmd: List[str], check: bool = True) -> Tuple[bool, str]:
        """Run a command and return success status and output"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr
        except Exception as e:
            return False, str(e)
    
    def get_gpu_count(self) -> int:
        """Get number of NVIDIA GPUs"""
        success, output = self.run_command(['nvidia-smi', '-L'])
        if success:
            return len([line for line in output.split('\n') if line.startswith('GPU')])
        return 0

    def get_driver_info(self) -> Tuple[str, str]:
        """Return (driver_version, cuda_version), querying the driver.

        Falls back to '?' for either field that cannot be determined."""
        driver = '?'
        cuda = '?'
        success, output = self.run_command(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'])
        if success and output.strip():
            driver = output.strip().split('\n')[0].strip()

        # CUDA version is only in the human-readable header, not --query-gpu.
        success, output = self.run_command(['nvidia-smi'])
        if success:
            match = re.search(r'CUDA Version:\s*([0-9.]+)', output)
            if match:
                cuda = match.group(1)
        return driver, cuda

    def get_live_status(self) -> List[Dict[str, Any]]:
        """Snapshot of live per-GPU telemetry for the dashboard/header."""
        cmd = ['nvidia-smi',
               '--query-gpu=index,temperature.gpu,fan.speed,utilization.gpu,'
               'clocks.gr,clocks.mem,power.draw,power.limit,memory.used,memory.total',
               '--format=csv,noheader,nounits']
        success, output = self.run_command(cmd)
        rows: List[Dict[str, Any]] = []
        if not success:
            return rows

        def _num(v):
            try:
                return int(float(v))
            except (ValueError, TypeError):
                return None

        for line in output.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 10:
                continue
            rows.append({
                'index': _num(parts[0]), 'temp': _num(parts[1]),
                'fan': _num(parts[2]), 'util': _num(parts[3]),
                'core_clock': _num(parts[4]), 'mem_clock': _num(parts[5]),
                'power_draw': _num(parts[6]), 'power_limit': _num(parts[7]),
                'mem_used': _num(parts[8]), 'mem_total': _num(parts[9]),
            })
        return rows

    def detect_gpus(self) -> List[Dict[str, Any]]:
        """Build a per-GPU capability lookup table from the driver.

        Queries nvidia-smi for each installed GPU's name, memory, power range
        and supported clock combinations. Falls back to hardcoded GPU_SPECS
        when the driver reports nothing usable, so menus can still offer sane
        options. Returns a list indexed by GPU id.
        """
        gpus: List[Dict[str, Any]] = []

        # One CSV row per GPU with the static identity/power fields.
        cmd = ['nvidia-smi',
               '--query-gpu=index,name,uuid,memory.total,'
               'power.min_limit,power.max_limit,power.default_limit',
               '--format=csv,noheader,nounits']
        success, output = self.run_command(cmd)
        if not success:
            return gpus

        def _to_int(value: str) -> Optional[int]:
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return None

        for line in output.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 7:
                continue

            index = _to_int(parts[0])
            if index is None:
                continue
            name = parts[1]
            gpu: Dict[str, Any] = {
                'index': index,
                'name': name,
                'uuid': parts[2],
                'memory_mb': _to_int(parts[3]),
                'power_min': _to_int(parts[4]),
                'power_max': _to_int(parts[5]),
                'power_default': _to_int(parts[6]),
                'mem_clocks': [],
                'graphics_clocks': {},
                'source': 'driver',
            }

            # Supported clock combinations for this specific GPU.
            ok, clk_out = self.run_command(
                ['nvidia-smi', '-i', str(index), '-q', '-d', 'SUPPORTED_CLOCKS'])
            if ok:
                mem_clocks, graphics_clocks = parse_supported_clocks(clk_out)
                gpu['mem_clocks'] = mem_clocks
                gpu['graphics_clocks'] = graphics_clocks

            # Hybrid fallback: fill gaps from the hardcoded table.
            spec = match_gpu_spec(name)
            if not gpu['mem_clocks'] and spec:
                gpu['source'] = 'fallback'
                gpu['mem_clocks'] = [spec['mem_max']]
                gpu['graphics_clocks'] = {spec['mem_max']: [spec['core_max']]}
            if gpu['power_max'] is None and spec:
                gpu['source'] = 'fallback'
                gpu['power_max'] = spec['power_max']

            gpus.append(gpu)

        return gpus
    
    def get_gpu_info(self) -> str:
        """Get current GPU information"""
        success, output = self.run_command(['nvidia-smi'])
        return output if success else "Failed to get GPU info"
    
    def set_persistence_mode(self, gpu_id: int, enabled: bool):
        """Enable/Disable persistence mode"""
        mode = '1' if enabled else '0'
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pm', mode]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_persistence'] = enabled
            self.save_settings()
            print(f"✓ Persistence mode {'enabled' if enabled else 'disabled'} for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set persistence mode: {output}")
        input("\nPress Enter to continue...")
    
    def set_power_limit(self, gpu_id: int, watts: int):
        """Set power limit for GPU"""
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pl', str(watts)]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_power_limit'] = watts
            self.save_settings()
            print(f"✓ Power limit set to {watts}W for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set power limit: {output}")
        input("\nPress Enter to continue...")
    
    def set_compute_mode(self, gpu_id: int, mode: int):
        """Set compute mode (0=Default, 1=Exclusive Thread, 2=Prohibited, 3=Exclusive Process)"""
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-c', str(mode)]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_compute_mode'] = mode
            self.save_settings()
            mode_names = {0: 'Default', 1: 'Exclusive Thread', 2: 'Prohibited', 3: 'Exclusive Process'}
            print(f"✓ Compute mode set to {mode_names.get(mode, 'Unknown')} for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set compute mode: {output}")
        input("\nPress Enter to continue...")
    
    def set_fan_speed(self, gpu_id: int, speed: int):
        """Set fan speed (requires coolbits enabled in X11)"""
        # Note: This requires X11 and coolbits configuration
        cmd = ['nvidia-settings', '-a', f'[gpu:{gpu_id}]/GPUFanControlState=1', '-a', f'[fan:{gpu_id}]/GPUTargetFanSpeed={speed}']
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_fan_speed'] = speed
            self.save_settings()
            print(f"✓ Fan speed set to {speed}% for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set fan speed (may require X11 and coolbits): {output}")
        input("\nPress Enter to continue...")

    # ------------------------------------------------------------------
    # lm-sensors / fancontrol (chassis / motherboard PWM fans)
    # ------------------------------------------------------------------
    def _command_exists(self, name: str) -> bool:
        """Check whether a command is on PATH"""
        return shutil.which(name) is not None

    def _read_sysfs(self, path: Optional[str]) -> Optional[str]:
        """Read a sysfs attribute, returning stripped text or None"""
        if not path:
            return None
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    def _sudo_write_sysfs(self, path: str, value: Any) -> Tuple[bool, str]:
        """Write to a root-owned sysfs attribute via sudo tee"""
        try:
            result = subprocess.run(
                ['sudo', 'tee', path],
                input=f'{value}\n', text=True, capture_output=True)
            return result.returncode == 0, result.stderr
        except Exception as e:
            return False, str(e)

    def _find_pwm_channels(self) -> List[Dict[str, Any]]:
        """Discover motherboard/chassis PWM fan channels via hwmon sysfs"""
        channels = []
        for hwmon in sorted(glob.glob('/sys/class/hwmon/hwmon*')):
            chip = self._read_sysfs(os.path.join(hwmon, 'name')) or os.path.basename(hwmon)
            for pwm_path in sorted(glob.glob(os.path.join(hwmon, 'pwm*'))):
                base = os.path.basename(pwm_path)
                if not re.fullmatch(r'pwm\d+', base):
                    continue  # skip pwmN_enable, pwmN_mode, etc.
                idx = base[3:]
                enable_path = os.path.join(hwmon, f'pwm{idx}_enable')
                fan_input = os.path.join(hwmon, f'fan{idx}_input')
                channels.append({
                    'chip': chip,
                    'index': idx,
                    'hwmon': hwmon,
                    'pwm': pwm_path,
                    'enable': enable_path if os.path.exists(enable_path) else None,
                    'fan_input': fan_input if os.path.exists(fan_input) else None,
                })
        return channels

    def _pwm_percent(self, raw: Optional[str]) -> str:
        """Convert a raw 0-255 PWM value to a percentage string"""
        if raw and raw.isdigit():
            return f"{round(int(raw) * 100 / 255)}%"
        return "?"

    def show_sensors(self):
        """Display current sensor readings via the `sensors` command"""
        os.system('clear')
        if not self._command_exists('sensors'):
            print("✗ 'sensors' not found. Install lm-sensors first (menu option).")
            input("\nPress Enter to continue...")
            return
        subprocess.run(['sensors'], check=False)
        input("\nPress Enter to continue...")

    def run_sensors_detect(self):
        """Run a safe automatic sensor probe to load Super I/O drivers"""
        os.system('clear')
        if not self._command_exists('sensors-detect'):
            print("✗ 'sensors-detect' not found. Install lm-sensors first (menu option).")
            input("\nPress Enter to continue...")
            return
        print("Running safe auto-probe: sudo sensors-detect --auto")
        print("This only performs the low-risk scans and writes detected")
        print("modules to /etc/modules (Debian) or /etc/modules-load.d.\n")
        subprocess.run(['sudo', 'sensors-detect', '--auto'], check=False)
        print("\nIf a chip was detected, load its driver now with:")
        print("  sudo modprobe <module>   (e.g. nct6775)")
        print("Newly loaded drivers expose pwmN files under /sys/class/hwmon.")
        input("\nPress Enter to continue...")

    def list_pwm_channels(self) -> List[Dict[str, Any]]:
        """Display discovered PWM fan channels with current state"""
        os.system('clear')
        print("=== PWM Fan Channels (lm-sensors) ===\n")
        channels = self._find_pwm_channels()
        if not channels:
            print("No PWM channels found.")
            print("Your motherboard's Super I/O sensor driver is likely not loaded.")
            print("Use 'Detect sensors (safe auto-probe)' then modprobe the driver.")
        else:
            print("enable: 0=no control  1=manual  2+=automatic (chip-specific)\n")
            for i, c in enumerate(channels):
                raw = self._read_sysfs(c['pwm'])
                en = self._read_sysfs(c['enable']) if c['enable'] else 'n/a'
                rpm = self._read_sysfs(c['fan_input']) if c['fan_input'] else 'n/a'
                print(f"{i}. {c['chip']} pwm{c['index']}: "
                      f"{self._pwm_percent(raw)} (raw {raw}), "
                      f"enable={en}, fan={rpm} RPM")
        input("\nPress Enter to continue...")
        return channels

    def _select_pwm_channel(self) -> Optional[Dict[str, Any]]:
        """Let the user pick a discovered PWM channel"""
        channels = self._find_pwm_channels()
        if not channels:
            print("\nNo PWM channels found. Run sensor detection first.")
            input("\nPress Enter to continue...")
            return None
        print("\nSelect fan channel:")
        for i, c in enumerate(channels):
            raw = self._read_sysfs(c['pwm'])
            print(f"{i}. {c['chip']} pwm{c['index']} ({self._pwm_percent(raw)})")
        try:
            choice = int(input("\nEnter channel: "))
            if 0 <= choice < len(channels):
                return channels[choice]
        except ValueError:
            pass
        print("Invalid choice")
        input("\nPress Enter to continue...")
        return None

    def set_fan_pwm(self, channel: Dict[str, Any], percent: int):
        """Set a PWM fan channel to a fixed percentage (manual mode)"""
        percent = max(0, min(100, percent))
        raw = round(percent * 255 / 100)
        if channel['enable']:
            ok, err = self._sudo_write_sysfs(channel['enable'], '1')
            if not ok:
                print(f"✗ Failed to enable manual mode: {err}")
                input("\nPress Enter to continue...")
                return
        ok, err = self._sudo_write_sysfs(channel['pwm'], raw)
        if ok:
            key = f"pwm|{channel['chip']}|{channel['index']}"
            self.settings[key] = percent
            self.save_settings()
            print(f"✓ Set {channel['chip']} pwm{channel['index']} to {percent}% (raw {raw})")
        else:
            print(f"✗ Failed to set PWM: {err}")
        input("\nPress Enter to continue...")

    def set_fan_auto(self, channel: Dict[str, Any]):
        """Return a PWM fan channel to automatic control"""
        if not channel['enable']:
            print("✗ This channel has no enable file; cannot set automatic mode.")
            input("\nPress Enter to continue...")
            return
        ok, err = self._sudo_write_sysfs(channel['enable'], '2')
        if not ok:  # some chips only support 0 (no control) as "hands off"
            ok, err = self._sudo_write_sysfs(channel['enable'], '0')
        if ok:
            self.settings.pop(f"pwm|{channel['chip']}|{channel['index']}", None)
            self.save_settings()
            print(f"✓ {channel['chip']} pwm{channel['index']} returned to automatic control")
        else:
            print(f"✗ Failed to set automatic mode: {err}")
        input("\nPress Enter to continue...")

    def apply_fan_settings(self):
        """Reapply saved PWM fan settings, matching by chip name and index"""
        channels = self._find_pwm_channels()
        applied = 0
        for key, percent in self.settings.items():
            if not key.startswith('pwm|'):
                continue
            try:
                _, chip, idx = key.split('|', 2)
            except ValueError:
                continue
            match = next((c for c in channels if c['chip'] == chip and c['index'] == idx), None)
            if not match:
                continue
            raw = round(int(percent) * 255 / 100)
            if match['enable']:
                self._sudo_write_sysfs(match['enable'], '1')
            ok, _ = self._sudo_write_sysfs(match['pwm'], raw)
            if ok:
                applied += 1
        if applied:
            print(f"✓ Reapplied {applied} PWM fan setting(s)")

    def install_lm_sensors(self):
        """Install lm-sensors and fancontrol using the system package manager"""
        os.system('clear')
        print("This installs lm-sensors (sensors, sensors-detect) and fancontrol.\n")
        if self._command_exists('apt-get'):
            cmd = ['sudo', 'apt-get', 'install', '-y', 'lm-sensors', 'fancontrol']
        elif self._command_exists('dnf'):
            cmd = ['sudo', 'dnf', 'install', '-y', 'lm_sensors']
        elif self._command_exists('pacman'):
            cmd = ['sudo', 'pacman', '-S', '--noconfirm', 'lm_sensors']
        else:
            print("✗ No supported package manager found (apt-get/dnf/pacman).")
            input("\nPress Enter to continue...")
            return
        print("Command:", ' '.join(cmd))
        if input("Proceed? (y/n): ").lower() == 'y':
            subprocess.run(cmd, check=False)
        input("\nPress Enter to continue...")

    def configure_fancontrol(self):
        """Launch pwmconfig to build /etc/fancontrol interactively"""
        os.system('clear')
        if not self._command_exists('pwmconfig'):
            print("✗ 'pwmconfig' not found. Install fancontrol first (menu option).")
            input("\nPress Enter to continue...")
            return
        if not self._find_pwm_channels():
            print("✗ No PWM channels detected. Run sensor detection first.")
            input("\nPress Enter to continue...")
            return
        print("pwmconfig is interactive: it briefly stops each fan to map")
        print("PWM outputs to fan tachometers, then writes /etc/fancontrol.")
        print("Press Ctrl+C inside it to abort safely.\n")
        if input("Launch pwmconfig now? (y/n): ").lower() == 'y':
            subprocess.run(['sudo', 'pwmconfig'], check=False)
        input("\nPress Enter to continue...")

    def manage_fancontrol_service(self):
        """Enable or disable the fancontrol systemd service"""
        os.system('clear')
        if not os.path.exists('/etc/fancontrol'):
            print("⚠ /etc/fancontrol not found. Run 'Configure fancontrol (pwmconfig)' first.\n")
        print("1. Enable and start fancontrol service")
        print("2. Disable and stop fancontrol service")
        print("3. Show fancontrol service status")
        choice = input("\nEnter choice: ").strip()
        if choice == '1':
            subprocess.run(['sudo', 'systemctl', 'enable', '--now', 'fancontrol'], check=False)
            print("✓ fancontrol enabled (starts on boot)")
        elif choice == '2':
            subprocess.run(['sudo', 'systemctl', 'disable', '--now', 'fancontrol'], check=False)
            print("✓ fancontrol disabled")
        elif choice == '3':
            subprocess.run(['systemctl', 'status', 'fancontrol', '--no-pager'], check=False)
        else:
            print("Invalid choice")
        input("\nPress Enter to continue...")

    def fan_control_menu(self):
        """Submenu for lm-sensors / fancontrol chassis fan control"""
        while True:
            os.system('clear')
            self._screen_title("SENSORS & FAN CONTROL", "lm-sensors / fancontrol")
            print()
            print(c("  SENSORS", 'bold', 'cyan'))
            print(f"   {c('[ 1]', 'yellow')}  Show sensor readings")
            print(f"   {c('[ 2]', 'yellow')}  Detect sensors (safe auto-probe)")
            print(f"   {c('[ 3]', 'yellow')}  Install lm-sensors / fancontrol")
            print()
            print(c("  MANUAL PWM FAN CONTROL", 'bold', 'cyan'))
            print(f"   {c('[ 4]', 'yellow')}  List PWM fan channels")
            print(f"   {c('[ 5]', 'yellow')}  Set fan speed (manual %)")
            print(f"   {c('[ 6]', 'yellow')}  Set fan channel to automatic")
            print()
            print(c("  AUTOMATIC FAN CURVES (fancontrol)", 'bold', 'cyan'))
            print(f"   {c('[ 7]', 'yellow')}  Configure fancontrol (pwmconfig)")
            print(f"   {c('[ 8]', 'yellow')}  Manage fancontrol service")
            print()
            print(c("   [ 0]", 'red') + c("  Back to main menu", 'red'))
            print(c("  " + "─" * 74, 'gray'))
            choice = input(c("\n  Select › ", 'bold', 'green')).strip()
            if choice == '0':
                break
            elif choice == '1':
                self.show_sensors()
            elif choice == '2':
                self.run_sensors_detect()
            elif choice == '3':
                self.install_lm_sensors()
            elif choice == '4':
                self.list_pwm_channels()
            elif choice == '5':
                channel = self._select_pwm_channel()
                if channel:
                    try:
                        percent = int(input("Enter fan speed (0-100%): "))
                        self.set_fan_pwm(channel, percent)
                    except ValueError:
                        print("Invalid input")
                        input("\nPress Enter to continue...")
            elif choice == '6':
                channel = self._select_pwm_channel()
                if channel:
                    self.set_fan_auto(channel)
            elif choice == '7':
                self.configure_fancontrol()
            elif choice == '8':
                self.manage_fancontrol_service()

    def enable_ecc(self, gpu_id: int, enabled: bool):
        """Enable/Disable ECC memory (requires reboot)"""
        mode = '1' if enabled else '0'
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-e', mode]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_ecc'] = enabled
            self.save_settings()
            print(f"✓ ECC {'enabled' if enabled else 'disabled'} for GPU {gpu_id} (requires reboot)")
        else:
            print(f"✗ Failed to set ECC mode: {output}")
        input("\nPress Enter to continue...")
    
    def set_auto_boost(self, gpu_id: int, enabled: bool):
        """Enable/Disable GPU auto boost"""
        mode = '1' if enabled else '0'
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '--auto-boost-default=' + mode]
        success, output = self.run_command(cmd, check=False)
        if success:
            self.settings[f'gpu_{gpu_id}_auto_boost'] = enabled
            self.save_settings()
            print(f"✓ Auto boost {'enabled' if enabled else 'disabled'} for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set auto boost (may not be supported): {output}")
        input("\nPress Enter to continue...")
    
    def set_application_clocks(self, gpu_id: int, mem_clock: int,
                               graphics_clock: int, pin: bool = False,
                               lock_memory: bool = False):
        """Lock the GPU core clock (and optionally the memory clock).

        Uses -lgc / -lmc (locked clocks), which are supported on GeForce cards.
        The legacy -ac 'application clocks' API is DEPRECATED on consumer GPUs,
        so it silently no-ops there.

        pin=False (default) sets a RANGE floor→target, so the card still idles
        down to its lowest clock when unused (lower idle power/temp) but is
        capped at `graphics_clock` under load. pin=True fixes it at exactly
        graphics_clock at all times (constant clock, higher idle draw).

        lock_memory=False (default) leaves the memory clock free so the card
        can reach its deep-idle power state. lock_memory=True pins memory to
        `mem_clock` (defeats the P2 compute downclock) but keeps the card out
        of deep idle — costing ~70W+ per card at idle. Only worth it on a card
        that is essentially always under load."""
        # Core / graphics clock lock — the primary control the user selected.
        # Cap uses a 0 lower bound so the card can still reach its deepest idle
        # state; only the ceiling is enforced. Pin fixes both bounds.
        floor = graphics_clock if pin else 0
        ok_gc, out_gc = self.run_command(
            ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-lgc',
             f'{floor},{graphics_clock}'], check=False)
        if ok_gc:
            self.settings[f'gpu_{gpu_id}_lock_gpu_clock'] = graphics_clock
            if pin:
                self.settings[f'gpu_{gpu_id}_lock_gpu_pin'] = True
            else:
                self.settings.pop(f'gpu_{gpu_id}_lock_gpu_pin', None)
            mode = (f"pinned at {graphics_clock} MHz" if pin
                    else f"capped at {graphics_clock} MHz (idles down normally)")
            print(f"✓ Core clock {mode} for GPU {gpu_id}")
        else:
            print(f"✗ Failed to lock core clock: {out_gc.strip()}")

        # Memory clock lock — opt-in only (raises idle power; see docstring).
        if lock_memory:
            ok_mc, out_mc = self.run_command(
                ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-lmc',
                 f'{mem_clock},{mem_clock}'], check=False)
            if ok_mc:
                self.settings[f'gpu_{gpu_id}_lock_mem_clock'] = mem_clock
                print(f"✓ Memory clock locked to {mem_clock} MHz for GPU {gpu_id} "
                      "(raises idle power)")
            else:
                print(f"⚠ Memory clock lock not applied "
                      f"({out_mc.strip() or 'unsupported'})")
        else:
            # Ensure no stale memory lock lingers for this GPU.
            self.run_command(
                ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rmc'], check=False)
            self.settings.pop(f'gpu_{gpu_id}_lock_mem_clock', None)

        self.save_settings()
        if ok_gc:
            print("  Verify with: nvidia-smi --query-gpu=clocks.gr "
                  "--format=csv,noheader")
        input("\nPress Enter to continue...")

    def reset_application_clocks(self, gpu_id: int):
        """Unlock core/memory clocks and clear any legacy application clocks."""
        ok_gc, _ = self.run_command(
            ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rgc'], check=False)
        self.run_command(
            ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rmc'], check=False)
        # Legacy reset; harmless no-op on cards where -ac is deprecated.
        self.run_command(
            ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rac'], check=False)
        for suffix in ('lock_gpu_clock', 'lock_gpu_pin', 'lock_mem_clock',
                       'app_mem_clock', 'app_graphics_clock'):
            self.settings.pop(f'gpu_{gpu_id}_{suffix}', None)
        self.save_settings()
        print(f"✓ Clocks unlocked / reset to default for GPU {gpu_id}")
        input("\nPress Enter to continue...")
    
    def set_gpu_reset(self, gpu_id: int):
        """Reset GPU (requires no processes using GPU)"""
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-r']
        success, output = self.run_command(cmd)
        if success:
            print(f"✓ GPU {gpu_id} reset successfully")
        else:
            print(f"✗ Failed to reset GPU: {output}")
        input("\nPress Enter to continue...")
    
    def set_accounting_mode(self, gpu_id: int, enabled: bool):
        """Enable/Disable accounting mode for process tracking"""
        mode = '1' if enabled else '0'
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-am', mode]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_accounting'] = enabled
            self.save_settings()
            print(f"✓ Accounting mode {'enabled' if enabled else 'disabled'} for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set accounting mode: {output}")
        input("\nPress Enter to continue...")
    
    def clear_accounting_data(self, gpu_id: int):
        """Clear all accounting data"""
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-caa']
        success, output = self.run_command(cmd)
        if success:
            print(f"✓ Accounting data cleared for GPU {gpu_id}")
        else:
            print(f"✗ Failed to clear accounting data: {output}")
        input("\nPress Enter to continue...")
    
    def set_mig_mode(self, gpu_id: int, enabled: bool):
        """Enable/Disable MIG mode (Multi-Instance GPU) - requires reboot"""
        mode = '1' if enabled else '0'
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-mig', mode]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_mig'] = enabled
            self.save_settings()
            print(f"✓ MIG mode {'enabled' if enabled else 'disabled'} for GPU {gpu_id} (requires reboot)")
        else:
            print(f"✗ Failed to set MIG mode (may not be supported): {output}")
        input("\nPress Enter to continue...")
    
    def query_gpu_details(self, gpu_id: int):
        """Query detailed GPU information"""
        cmd = ['nvidia-smi', '-i', str(gpu_id), '-q']
        success, output = self.run_command(cmd)
        if success:
            print(output)
        else:
            print(f"✗ Failed to query GPU details: {output}")
        input("\nPress Enter to continue...")
    
    def monitor_gpu_live(self):
        """Live monitoring using dmon"""
        print("Starting live GPU monitoring (Press Ctrl+C to stop)...")
        print("Legend: pwr=power, temp=temperature, sm=streaming multiprocessor, mem=memory, enc=encoder, dec=decoder")
        print()
        try:
            subprocess.run(['nvidia-smi', 'dmon', '-s', 'pucmvet'], check=False)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
        input("\nPress Enter to continue...")
    
    def monitor_processes(self):
        """Monitor GPU processes"""
        print("Monitoring GPU processes (Press Ctrl+C to stop)...")
        print()
        try:
            subprocess.run(['nvidia-smi', 'pmon', '-s', 'um'], check=False)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
        input("\nPress Enter to continue...")
    
    def query_supported_clocks(self, gpu_id: int):
        """Query supported clock combinations"""
        cmd = ['nvidia-smi', '-i', str(gpu_id), '-q', '-d', 'SUPPORTED_CLOCKS']
        success, output = self.run_command(cmd)
        if success:
            print(output)
        else:
            print(f"✗ Failed to query supported clocks: {output}")
        input("\nPress Enter to continue...")
    
    def set_gom_mode(self, gpu_id: int, mode: str):
        """Set GPU Operation Mode (all-on, compute, low-dp)"""
        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '--gom', mode]
        success, output = self.run_command(cmd)
        if success:
            self.settings[f'gpu_{gpu_id}_gom'] = mode
            self.save_settings()
            print(f"✓ GOM mode set to {mode} for GPU {gpu_id}")
        else:
            print(f"✗ Failed to set GOM mode: {output}")
        input("\nPress Enter to continue...")
    
    def apply_all_settings(self):
        """Apply all saved settings"""
        if not self.settings:
            print("No saved settings to apply")
            time.sleep(2)
            return
            
        print("Applying saved settings...")
        applied_count = 0
        total_count = 0
        
        for key, value in self.settings.items():
            parts = key.split('_')
            if len(parts) >= 3 and parts[0] == 'gpu':
                gpu_id = int(parts[1])
                setting_type = '_'.join(parts[2:])
                total_count += 1
                
                try:
                    if setting_type == 'persistence':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pm', '1' if value else '0']
                        success, _ = self.run_command(cmd)
                        if success: applied_count += 1
                    elif setting_type == 'power_limit':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pl', str(value)]
                        success, _ = self.run_command(cmd)
                        if success: applied_count += 1
                    elif setting_type == 'compute_mode':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-c', str(value)]
                        success, _ = self.run_command(cmd)
                        if success: applied_count += 1
                    elif setting_type == 'ecc':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-e', '1' if value else '0']
                        success, _ = self.run_command(cmd, check=False)
                        if success: applied_count += 1
                    elif setting_type == 'lock_gpu_pin':
                        # Marker flag consumed by lock_gpu_clock; not a command.
                        total_count -= 1
                    elif setting_type == 'lock_gpu_clock':
                        pinned = self.settings.get(f'gpu_{gpu_id}_lock_gpu_pin', False)
                        lo = value if pinned else 0
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-lgc',
                               f'{lo},{value}']
                        success, _ = self.run_command(cmd, check=False)
                        if success: applied_count += 1
                    elif setting_type == 'lock_mem_clock':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-lmc',
                               f'{value},{value}']
                        success, _ = self.run_command(cmd, check=False)
                        if success: applied_count += 1
                    elif setting_type == 'accounting':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-am', '1' if value else '0']
                        success, _ = self.run_command(cmd, check=False)
                        if success: applied_count += 1
                    elif setting_type == 'mig':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-mig', '1' if value else '0']
                        success, _ = self.run_command(cmd, check=False)
                        if success: applied_count += 1
                    elif setting_type == 'gom':
                        cmd = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '--gom', str(value)]
                        success, _ = self.run_command(cmd, check=False)
                        if success: applied_count += 1
                except Exception as e:
                    print(f"Failed to apply {key}: {e}")

        # Reapply lm-sensors PWM fan settings (keyed as pwm|chip|index)
        self.apply_fan_settings()

        print(f"✓ Applied {applied_count}/{total_count} settings successfully")
        if applied_count < total_count:
            print("Some settings failed - this is normal for unsupported features")
        time.sleep(2)
    
    def reset_all_gpus(self):
        """Reset all GPUs to default settings"""
        print("Resetting all GPUs to defaults...")
        for gpu_id in range(self.gpu_count):
            # Reset clocks
            subprocess.run(['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rac'], capture_output=True)
            # Reset power limit
            subprocess.run(['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pl', '0'], capture_output=True)
            # Set persistence mode off
            subprocess.run(['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pm', '0'], capture_output=True)
        
        # Clear settings
        self.settings = {}
        self.save_settings()
        print("✓ All GPUs reset to defaults")
        input("\nPress Enter to continue...")
    
    def show_current_settings(self):
        """Display current saved settings"""
        os.system('clear')
        print("=== Current Saved Settings ===\n")
        if not self.settings:
            print("No saved settings")
        else:
            for key, value in sorted(self.settings.items()):
                print(f"{key}: {value}")
        input("\nPress Enter to continue...")
    
    def gpu_label(self, gpu_id: int) -> str:
        """Human-readable label for a GPU from the lookup table."""
        if 0 <= gpu_id < len(self.gpus):
            gpu = self.gpus[gpu_id]
            name = gpu.get('name') or f'GPU {gpu_id}'
            mem_mb = gpu.get('memory_mb')
            if mem_mb:
                return f"{name} ({round(mem_mb / 1024)}GB)"
            return name
        return f"GPU {gpu_id}"

    def show_gpu_capabilities(self):
        """Display the detected per-GPU capability lookup table."""
        os.system('clear')
        print("=== Detected GPU Capabilities ===\n")
        if not self.gpus:
            print("No NVIDIA GPUs detected!")
            input("\nPress Enter to continue...")
            return

        for gpu in self.gpus:
            print(f"GPU {gpu['index']}: {gpu.get('name', 'Unknown')}")
            if gpu.get('uuid'):
                print(f"  UUID:        {gpu['uuid']}")
            if gpu.get('memory_mb'):
                print(f"  Memory:      {gpu['memory_mb']} MB")

            pmin = gpu.get('power_min')
            pmax = gpu.get('power_max')
            pdef = gpu.get('power_default')
            if pmax is not None:
                rng = f"{pmin if pmin is not None else '?'}-{pmax} W"
                if pdef is not None:
                    rng += f" (default {pdef} W)"
                print(f"  Power:       {rng}")

            mem_clocks = gpu.get('mem_clocks') or []
            if mem_clocks:
                print(f"  Mem clocks:  {', '.join(str(m) for m in mem_clocks)} MHz")
                graphics_clocks = gpu.get('graphics_clocks') or {}
                top_mem = mem_clocks[0]
                gc = graphics_clocks.get(top_mem) or []
                if gc:
                    shown = ', '.join(str(g) for g in gc[:8])
                    more = ' ...' if len(gc) > 8 else ''
                    print(f"  Graphics @ {top_mem}: {shown}{more} MHz")
            else:
                print("  Clocks:      not reported by driver")

            if gpu.get('source') == 'fallback':
                print("  (values approximate - from built-in fallback table)")
            print()

        input("Press Enter to continue...")

    def select_gpu(self) -> Optional[int]:
        """Let user select a GPU"""
        if self.gpu_count == 0:
            print("No NVIDIA GPUs detected!")
            input("\nPress Enter to continue...")
            return None
        
        print("\nSelect GPU:")
        for i in range(self.gpu_count):
            print(f"{i}. {self.gpu_label(i)}")
        print(f"{self.gpu_count}. All GPUs")
        
        try:
            choice = int(input("\nEnter choice: "))
            if 0 <= choice <= self.gpu_count:
                return choice if choice < self.gpu_count else -1  # -1 for all GPUs
        except ValueError:
            pass
        
        print("Invalid choice")
        input("\nPress Enter to continue...")
        return None
    
    @staticmethod
    def _print_numbered(items: List[int], unit: str = "MHz"):
        """Print a numbered pick-list, using columns when the list is long."""
        n = len(items)
        if n <= 12:
            for i, v in enumerate(items):
                print(f"  {i}. {v} {unit}")
            return
        # Compact multi-column layout for long lists (e.g. 100+ clocks).
        cols = 4
        rows = (n + cols - 1) // cols
        width = max(len(f"{i}. {v}") for i, v in enumerate(items)) + 2
        for r in range(rows):
            cells = []
            for col in range(cols):
                idx = col * rows + r
                if idx < n:
                    cells.append(f"{idx}. {items[idx]}".ljust(width))
            print("  " + "".join(cells).rstrip())
        print(f"  (values in {unit})")

    def prompt_supported_clocks(self, gpu_id: int) -> Optional[Tuple[int, int]]:
        """Let the user pick a memory/graphics clock pair from what the GPU
        actually supports. Returns (mem_clock, graphics_clock) or None if
        cancelled. Falls back to manual entry when no data is available or the
        user chooses the manual option."""
        gpu = self.gpus[gpu_id] if 0 <= gpu_id < len(self.gpus) else {}
        mem_clocks = gpu.get('mem_clocks') or []
        graphics_clocks = gpu.get('graphics_clocks') or {}

        if gpu.get('source') == 'fallback':
            print("\n(Note: supported clocks below are approximate fallback values)")

        if not mem_clocks:
            print("\nNo supported-clock data available for this GPU; entering values manually.")
            return self._manual_clock_entry()

        # Memory clock defaults to the GPU's maximum; most tuning is about the
        # graphics (core) clock, so that is the primary choice below. The user
        # can switch to a different memory clock with the 'm' option first.
        mem_clock = mem_clocks[0]
        print(f"\nMemory clock defaults to {mem_clock} MHz (max).")
        if len(mem_clocks) > 1:
            change = input("Change memory clock? (y/n): ").strip().lower()
            if change == 'y':
                print(f"\nSupported memory clocks for {self.gpu_label(gpu_id)}:")
                self._print_numbered(mem_clocks)
                print(f"  {len(mem_clocks)}. Enter manually")
                try:
                    m_choice = int(input("Select memory clock: "))
                except ValueError:
                    print("Invalid choice")
                    return None
                if m_choice == len(mem_clocks):
                    return self._manual_clock_entry()
                if not 0 <= m_choice < len(mem_clocks):
                    print("Invalid choice")
                    return None
                mem_clock = mem_clocks[m_choice]

        # Primary choice: graphics (core) clock for the selected memory clock.
        gc_list = graphics_clocks.get(mem_clock) or []
        if not gc_list:
            print("No graphics clocks listed for that memory clock; enter manually.")
            g_manual = self._manual_clock_entry()
            return (mem_clock, g_manual[1]) if g_manual else None

        print(f"\nSupported graphics (core) clocks at {mem_clock} MHz memory:")
        self._print_numbered(gc_list)
        print(f"  {len(gc_list)}. Enter manually")
        try:
            g_choice = int(input("Select graphics (core) clock: "))
        except ValueError:
            print("Invalid choice")
            return None
        if g_choice == len(gc_list):
            g_manual = self._manual_clock_entry()
            return (mem_clock, g_manual[1]) if g_manual else None
        if not 0 <= g_choice < len(gc_list):
            print("Invalid choice")
            return None
        return mem_clock, gc_list[g_choice]

    def _manual_clock_entry(self) -> Optional[Tuple[int, int]]:
        """Free-text fallback for entering a mem/graphics clock pair."""
        try:
            mem_clock = int(input("Enter memory clock (MHz): "))
            graphics_clock = int(input("Enter graphics clock (MHz): "))
            return mem_clock, graphics_clock
        except ValueError:
            print("Invalid input")
            return None

    def prompt_power_limit(self, gpu_id: int) -> Optional[int]:
        """Prompt for a power limit, showing and validating against the GPU's
        real min/max range. Returns watts or None if invalid/cancelled."""
        gpu = self.gpus[gpu_id] if 0 <= gpu_id < len(self.gpus) else {}
        pmin = gpu.get('power_min')
        pmax = gpu.get('power_max')
        pdef = gpu.get('power_default')

        if pmax is not None:
            rng = f"{pmin if pmin is not None else '?'}-{pmax} W"
            if pdef is not None:
                rng += f" (default {pdef} W)"
            print(f"\nAllowed power range for {self.gpu_label(gpu_id)}: {rng}")
        try:
            watts = int(input("Enter power limit in watts: "))
        except ValueError:
            print("Invalid input")
            return None
        if pmin is not None and watts < pmin:
            print(f"✗ {watts}W is below the minimum ({pmin}W)")
            return None
        if pmax is not None and watts > pmax:
            print(f"✗ {watts}W exceeds the maximum ({pmax}W)")
            return None
        return watts

    # ------------------------------------------------------------------
    # Menu action helpers
    # ------------------------------------------------------------------
    def _for_selected(self, gpu_id: int, action):
        """Run action(i) for a single GPU, or every GPU when gpu_id == -1."""
        if gpu_id == -1:
            for i in range(self.gpu_count):
                action(i)
        else:
            action(gpu_id)

    def _pause(self):
        input(c("\nPress Enter to continue...", 'dim'))

    def action_show_info(self):
        os.system('clear')
        print(self.get_gpu_info())
        self._pause()

    def action_query_details(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None and gpu_id != -1:
            self.query_gpu_details(gpu_id)

    def action_query_supported_clocks(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None and gpu_id != -1:
            self.query_supported_clocks(gpu_id)

    def action_set_persistence(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            enabled = input("Enable persistence mode? (y/n): ").lower() == 'y'
            self._for_selected(gpu_id, lambda i: self.set_persistence_mode(i, enabled))

    def action_set_power_limit(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            watts = self.prompt_power_limit(gpu_id if gpu_id != -1 else 0)
            if watts is not None:
                self._for_selected(gpu_id, lambda i: self.set_power_limit(i, watts))
            else:
                self._pause()

    def action_set_app_clocks(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            clocks = self.prompt_supported_clocks(gpu_id if gpu_id != -1 else 0)
            if clocks is not None:
                mem_clock, graphics_clock = clocks
                print("\n  Lock mode:")
                print(f"    {c('[c]', 'yellow')} Cap max — idles down when unused "
                      "(lower idle power/temp) [recommended]")
                print(f"    {c('[p]', 'yellow')} Pin exact — constant clock, "
                      "higher idle draw")
                pin = input("  Choose [c]: ").strip().lower().startswith('p')
                print(f"\n  Also lock memory clock to {mem_clock} MHz? This defeats "
                      "the P2\n  compute downclock but keeps the card out of deep "
                      "idle (~70W+/card).")
                lock_memory = input("  Lock memory? (y/N): ").strip().lower().startswith('y')
                self._for_selected(
                    gpu_id,
                    lambda i: self.set_application_clocks(
                        i, mem_clock, graphics_clock, pin=pin, lock_memory=lock_memory))
            else:
                self._pause()

    def action_reset_app_clocks(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            self._for_selected(gpu_id, self.reset_application_clocks)

    def action_set_gom(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            print("\nGPU Operation Modes:")
            print("  1. all-on   (all features enabled)")
            print("  2. compute  (compute only, no graphics)")
            print("  3. low-dp   (low double precision)")
            mode_map = {'1': 'all-on', '2': 'compute', '3': 'low-dp'}
            mode_choice = input("Enter mode choice: ")
            if mode_choice in mode_map:
                self._for_selected(gpu_id, lambda i: self.set_gom_mode(i, mode_map[mode_choice]))
            else:
                print("Invalid choice")
                self._pause()

    def action_set_auto_boost(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            enabled = input("Enable auto boost? (y/n): ").lower() == 'y'
            self._for_selected(gpu_id, lambda i: self.set_auto_boost(i, enabled))

    def action_set_compute_mode(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            print("\nCompute Modes:")
            print("  0. Default")
            print("  1. Exclusive Thread")
            print("  2. Prohibited")
            print("  3. Exclusive Process")
            try:
                mode = int(input("Enter mode: "))
            except ValueError:
                print("Invalid input")
                self._pause()
                return
            if 0 <= mode <= 3:
                self._for_selected(gpu_id, lambda i: self.set_compute_mode(i, mode))
            else:
                print("Invalid mode")
                self._pause()

    def action_toggle_ecc(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            enabled = input("Enable ECC memory? (y/n): ").lower() == 'y'
            self._for_selected(gpu_id, lambda i: self.enable_ecc(i, enabled))

    def action_toggle_mig(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            enabled = input("Enable MIG mode? (y/n): ").lower() == 'y'
            self._for_selected(gpu_id, lambda i: self.set_mig_mode(i, enabled))

    def action_toggle_accounting(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            enabled = input("Enable accounting mode? (y/n): ").lower() == 'y'
            self._for_selected(gpu_id, lambda i: self.set_accounting_mode(i, enabled))

    def action_clear_accounting(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            self._for_selected(gpu_id, self.clear_accounting_data)

    def action_set_fan_speed(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            try:
                speed = int(input("Enter fan speed (0-100%): "))
            except ValueError:
                print("Invalid input")
                self._pause()
                return
            if 0 <= speed <= 100:
                self._for_selected(gpu_id, lambda i: self.set_fan_speed(i, speed))
            else:
                print("Speed must be 0-100")
                self._pause()

    def action_reset_gpu(self):
        gpu_id = self.select_gpu()
        if gpu_id is not None:
            if input("Are you sure you want to reset GPU? (y/n): ").lower() == 'y':
                self._for_selected(gpu_id, self.set_gpu_reset)

    def action_reset_all(self):
        if input("Are you sure you want to reset ALL GPUs to defaults? (y/n): ").lower() == 'y':
            self.reset_all_gpus()

    # ------------------------------------------------------------------
    # vLLM optimizer (integrated, multi-GPU / tensor-parallel aware)
    # ------------------------------------------------------------------
    def _vllm_select_gpus(self) -> Optional[List[int]]:
        """Choose which GPUs to configure for vLLM. Defaults to all (the
        common tensor-parallel case). Returns a list of indices or None."""
        if self.gpu_count == 0:
            print(c("  No GPUs detected.", 'red'))
            self._pause()
            return None
        if self.gpu_count == 1:
            return [0]
        print("\n  Which GPUs? (tensor-parallel usually uses all)")
        print(f"    default = all {self.gpu_count}: 0-{self.gpu_count - 1}")
        raw = input("  Enter 'all' or comma list [all]: ").strip().lower()
        if raw in ('', 'all'):
            return list(range(self.gpu_count))
        try:
            ids = sorted({int(x) for x in raw.replace(' ', '').split(',')})
        except ValueError:
            print(c("  Invalid list", 'red'))
            self._pause()
            return None
        if any(i < 0 or i >= self.gpu_count for i in ids) or not ids:
            print(c("  Index out of range", 'red'))
            self._pause()
            return None
        return ids

    def _vllm_power_target(self, gpu: Dict[str, Any], pct: int) -> Optional[int]:
        """Compute a power-limit target as pct of max, clamped to the GPU's
        allowed [min, max] range."""
        pmax = gpu.get('power_max')
        if pmax is None:
            return None
        pmin = gpu.get('power_min') or 0
        target = int(round(pmax * pct / 100))
        return max(pmin, min(pmax, target))

    def apply_vllm_profile(self, gpu_ids: List[int], key: str):
        """Apply a vLLM profile across the selected GPUs and persist the
        settings using the panel's existing key convention so they survive
        via 'Apply all saved settings' / the systemd service."""
        profile = VLLM_PROFILES[key]
        os.system('clear')
        self._screen_title(f"APPLYING · {profile['name'].upper()}",
                           f"GPUs {', '.join(map(str, gpu_ids))}")
        print()

        for gpu_id in gpu_ids:
            gpu = self.gpus[gpu_id]
            results = []

            # Persistence mode (always on for serving).
            ok, _ = self.run_command(
                ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pm', '1'])
            if ok:
                self.settings[f'gpu_{gpu_id}_persistence'] = True
            results.append(('Persistence on', ok))

            # Compute mode Default (0) — what vLLM wants.
            ok, _ = self.run_command(
                ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-c', '0'])
            if ok:
                self.settings[f'gpu_{gpu_id}_compute_mode'] = 0
            results.append(('Compute mode: Default', ok))

            # Power limit (clamped).
            watts = self._vllm_power_target(gpu, profile['power_pct'])
            if watts is not None:
                ok, _ = self.run_command(
                    ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pl', str(watts)],
                    check=False)
                if ok:
                    self.settings[f'gpu_{gpu_id}_power_limit'] = watts
                results.append((f'Power limit {watts}W '
                                f'({profile["power_pct"]}% of max)', ok))

            # Clocks: lock to max for perf profiles (via -lgc/-lmc, which work
            # on GeForce; -ac is deprecated there), otherwise unlock so the
            # boost algorithm can manage them (better perf/watt).
            if profile['lock_max_clocks'] and gpu.get('mem_clocks'):
                mem = gpu['mem_clocks'][0]
                gc = (gpu.get('graphics_clocks') or {}).get(mem) or []
                if gc:
                    graphics = gc[0]
                    # Cap max core clock (0→max) so cards still idle down between
                    # requests. Memory is left unlocked so cards can reach deep
                    # idle; the P2 downclock under load costs only ~2.5%.
                    ok, _ = self.run_command(
                        ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-lgc',
                         f'0,{graphics}'], check=False)
                    if ok:
                        self.settings[f'gpu_{gpu_id}_lock_gpu_clock'] = graphics
                        self.settings.pop(f'gpu_{gpu_id}_lock_gpu_pin', None)
                    results.append((f'Cap core clock ≤{graphics}MHz', ok))
            else:
                ok, _ = self.run_command(
                    ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rgc'], check=False)
                self.run_command(
                    ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rmc'], check=False)
                for suffix in ('lock_gpu_clock', 'lock_gpu_pin', 'lock_mem_clock',
                               'app_mem_clock', 'app_graphics_clock'):
                    self.settings.pop(f'gpu_{gpu_id}_{suffix}', None)
                results.append(('Clocks: default (boost-managed)', ok))

            label = self.gpu_label(gpu_id)
            print(c(f"  GPU {gpu_id} — {label}", 'bold', 'cyan'))
            for text, ok in results:
                mark = c('✓', 'green') if ok else c('✗', 'red')
                print(f"    {mark} {text}")
            print()

        self.save_settings()
        print(c("  Settings saved — reapply anytime via 'Apply all saved settings'.", 'dim'))

        self._vllm_show_launch(gpu_ids, key)
        self._pause()

    def _vllm_launch_plan(self, gpu_ids: List[int], key: str) -> Dict[str, Any]:
        """Compute vLLM launch parameters for the selected GPUs."""
        profile = VLLM_PROFILES[key]
        names = {self.gpus[i].get('name', '') for i in gpu_ids}
        vrams = [self.gpus[i].get('memory_mb') or 0 for i in gpu_ids]
        min_vram_gb = (min(vrams) / 1024) if vrams else 0
        model = next(iter(names)) if names else ''

        # Context / batch heuristics keyed off per-GPU VRAM and model family.
        if 'A100' in model or 'H100' in model:
            max_len, base_seqs = (32768, 256) if min_vram_gb >= 79 else (16384, 128)
        elif any(t in model for t in ('A6000', 'RTX 6000', 'A40')):
            max_len, base_seqs = 16384, 128
        elif any(t in model for t in ('4090', '4080')):
            max_len, base_seqs = 8192, 64
        elif any(t in model for t in ('3090', 'A5000')) or min_vram_gb >= 24:
            max_len, base_seqs = 8192, 48
        elif min_vram_gb >= 16:
            max_len, base_seqs = 4096, 16
        else:
            max_len, base_seqs = 2048, 8

        n = len(gpu_ids)
        # Tensor parallelism multiplies effective KV-cache capacity → scale
        # concurrency with GPU count for throughput-oriented profiles.
        seqs = base_seqs * n if key == 'max_throughput' else base_seqs
        flags = [
            f'--tensor-parallel-size {n}',
            f'--gpu-memory-utilization {profile["mem_util"]}',
            f'--max-model-len {max_len}',
            f'--max-num-seqs {seqs}',
        ] + profile['extra_flags']

        return {
            'devices': ','.join(map(str, gpu_ids)),
            'tp': n, 'flags': flags,
            'min_vram_gb': min_vram_gb, 'mixed': len(names) > 1,
            'total_vram_gb': sum(vrams) / 1024 if vrams else 0,
        }

    def _vllm_show_launch(self, gpu_ids: List[int], key: str):
        plan = self._vllm_launch_plan(gpu_ids, key)
        print()
        print(c("  ── Suggested vLLM launch " + "─" * 49, 'gray'))
        if plan['mixed']:
            print(c("  ⚠ Selected GPUs are not identical; sized to the smallest.",
                    'yellow'))
        print(c(f"  {plan['tp']}-way tensor parallel · "
                f"~{plan['total_vram_gb']:.0f}GB total "
                f"({plan['min_vram_gb']:.0f}GB/GPU)", 'dim'))
        print()
        line_col = 'green'
        print(c(f"  CUDA_VISIBLE_DEVICES={plan['devices']} \\", line_col))
        print(c("  vllm serve <MODEL> \\", line_col))
        for i, fl in enumerate(plan['flags']):
            tail = ' \\' if i < len(plan['flags']) - 1 else ''
            print(c(f"    {fl}{tail}", line_col))
        print()
        print(c("  Env: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", 'dim'))
        if plan['tp'] > 1:
            print(c("  Note: --tensor-parallel-size must evenly divide the model's "
                    "attention heads.", 'dim'))

    def vllm_status(self):
        """Show live status + a vLLM-readiness check for all GPUs."""
        os.system('clear')
        self._screen_title("vLLM READINESS", f"{self.gpu_count} GPU(s)")
        print()
        rows = self.get_live_status()
        for r in rows:
            idx = r['index']
            print(c(f"  GPU {idx} — {self.gpu_label(idx)}", 'bold', 'cyan'))
            checks = []
            pl, pmax = r.get('power_limit'), self.gpus[idx].get('power_max')
            if pl and pmax and pl >= pmax * 0.85:
                checks.append((True, f"Power limit near max ({pl}/{pmax}W)"))
            else:
                checks.append((False, f"Power limit not maximized ({pl}W)"))
            temp = r.get('temp') or 0
            checks.append((temp < 80, f"Temp {temp}°C"))
            for ok, text in checks:
                mark = c('✓', 'green') if ok else c('⚠', 'yellow')
                print(f"    {mark} {text}")
            print()
        print(c("  Tip: tensor-parallel serving wants persistence ON and identical "
                "power/clock settings across all GPUs.", 'dim'))
        self._pause()

    def vllm_reset(self):
        gpu_ids = self._vllm_select_gpus()
        if not gpu_ids:
            return
        if input("  Reset these GPUs to defaults? (y/n): ").lower() != 'y':
            return
        for i in gpu_ids:
            self.run_command(['sudo', 'nvidia-smi', '-i', str(i), '-rgc'], check=False)
            self.run_command(['sudo', 'nvidia-smi', '-i', str(i), '-rmc'], check=False)
            self.run_command(['sudo', 'nvidia-smi', '-i', str(i), '-rac'], check=False)
            self.run_command(['sudo', 'nvidia-smi', '-i', str(i), '-pl', '0'], check=False)
            for suffix in ('lock_gpu_clock', 'lock_mem_clock',
                           'app_mem_clock', 'app_graphics_clock', 'power_limit'):
                self.settings.pop(f'gpu_{i}_{suffix}', None)
        self.save_settings()
        print(c(f"  ✓ Reset GPUs {', '.join(map(str, gpu_ids))} to defaults", 'green'))
        self._pause()

    def action_launch_vllm(self):
        """Integrated vLLM optimizer submenu."""
        while True:
            os.system('clear')
            gpu_names = {g.get('name', '') for g in self.gpus}
            summary = (f"{self.gpu_count}× {next(iter(gpu_names))}"
                       if len(gpu_names) == 1 and self.gpu_count
                       else f"{self.gpu_count} GPU(s)")
            self._screen_title("vLLM OPTIMIZER", summary)
            print()
            print(c("  OPTIMIZATION PROFILES", 'bold', 'cyan'))
            keys = list(VLLM_PROFILES.keys())
            for i, k in enumerate(keys, 1):
                p = VLLM_PROFILES[k]
                print(f"   {c(f'[{i}]', 'yellow')}  {c(p['name'], 'bold')}")
                print(f"        {c(p['description'], 'dim')}")
            print()
            print(c("  TOOLS", 'bold', 'cyan'))
            print(f"   {c('[6]', 'yellow')}  vLLM readiness check")
            print(f"   {c('[7]', 'yellow')}  Generate launch command (no changes)")
            print(f"   {c('[8]', 'yellow')}  Reset GPUs to defaults")
            print()
            print(c("   [0]", 'red') + c("  Back to main menu", 'red'))
            print(c("  " + "─" * 74, 'gray'))
            choice = input(c("\n  Select › ", 'bold', 'green')).strip()

            if choice == '0':
                break
            elif choice in {str(i) for i in range(1, len(keys) + 1)}:
                key = keys[int(choice) - 1]
                gpu_ids = self._vllm_select_gpus()
                if gpu_ids:
                    self.apply_vllm_profile(gpu_ids, key)
            elif choice == '6':
                self.vllm_status()
            elif choice == '7':
                gpu_ids = self._vllm_select_gpus()
                if gpu_ids:
                    print("\n  Profile for launch params:")
                    for i, k in enumerate(keys, 1):
                        print(f"    {i}. {VLLM_PROFILES[k]['name']}")
                    pk = input("  Profile [1]: ").strip() or '1'
                    if pk in {str(i) for i in range(1, len(keys) + 1)}:
                        self._vllm_show_launch(gpu_ids, keys[int(pk) - 1])
                        self._pause()
            elif choice == '8':
                self.vllm_reset()
            else:
                print(c("  Invalid choice", 'red'))
                time.sleep(0.8)

    def live_dashboard(self, interval: float = 1.0):
        """Full-screen auto-refreshing GPU dashboard. Ctrl+C to return."""
        # Hide cursor for a cleaner refresh; always restore it on the way out.
        if Style.enabled:
            sys.stdout.write('\033[?25l')
        try:
            while True:
                os.system('clear')
                self._screen_title(
                    "LIVE GPU DASHBOARD",
                    f"Driver {self.driver_version}   CUDA {self.cuda_version}   •   "
                    f"refresh {interval:g}s   •   {time.strftime('%H:%M:%S')}")
                print()

                rows = self.get_live_status()
                if not rows:
                    print(c("  No GPU telemetry available.", 'red'))
                else:
                    header = (f"  {'GPU':<3} {'NAME':<20} {'TEMP':>5} {'FAN':>4} "
                              f"{'UTIL':>5} {'CORE':>8} {'MEM CLK':>8} "
                              f"{'POWER':>11} {'MEMORY':>13}")
                    print(c(header, 'bold', 'gray'))
                    print(c("  " + "─" * 84, 'gray'))
                    for r in rows:
                        print("  " + self._format_status_row(r, wide=True))

                    # Aggregate footer line.
                    draws = [r['power_draw'] for r in rows if r['power_draw'] is not None]
                    lims = [r['power_limit'] for r in rows if r['power_limit'] is not None]
                    used = [r['mem_used'] for r in rows if r['mem_used'] is not None]
                    tot = [r['mem_total'] for r in rows if r['mem_total'] is not None]
                    if draws and lims:
                        print(c("  " + "─" * 84, 'gray'))
                        agg = (f"  {'TOTAL':<20}"
                               f"{sum(draws)}/{sum(lims)}W".rjust(46) +
                               f"   {sum(used)/1024:.1f}/{sum(tot)/1024:.0f}GB".rjust(15))
                        print(c(agg, 'bold', 'white'))

                print(c("\n  Press Ctrl+C to return to the menu", 'dim'))
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            if Style.enabled:
                sys.stdout.write('\033[?25h')  # restore cursor
                sys.stdout.flush()

    def _format_status_row(self, r: Dict[str, Any], wide: bool = False) -> str:
        """Render one live-status row (used by menu header and dashboard)."""
        idx = r['index']
        name = (self.gpus[idx]['name'] if idx is not None and idx < len(self.gpus)
                else f"GPU {idx}")
        name = name.replace('NVIDIA ', '')
        temp = r.get('temp')
        temp_s = f"{temp}°C" if temp is not None else "  -"
        temp_col = 'red' if (temp or 0) >= 80 else 'yellow' if (temp or 0) >= 65 else 'green'
        util = r.get('util')
        util_s = f"{util}%" if util is not None else " -"
        pdraw, plim = r.get('power_draw'), r.get('power_limit')
        power_s = f"{pdraw}/{plim}W" if pdraw is not None and plim is not None else "   -"
        mused, mtot = r.get('mem_used'), r.get('mem_total')
        mem_s = (f"{mused/1024:.1f}/{mtot/1024:.0f}GB"
                 if mused is not None and mtot is not None else "   -")

        if not wide:
            name = name[:26]
            return (f"{c(f'{idx:<3}', 'bold', 'cyan')} {name:<26} "
                    f"{c(f'{temp_s:>5}', temp_col)} {util_s:>5} {power_s:>11} {mem_s:>13}")

        name = name[:20]
        fan = r.get('fan')
        fan_s = f"{fan}%" if fan is not None else "  -"
        core = r.get('core_clock')
        core_s = f"{core}MHz" if core is not None else "   -"
        memclk = r.get('mem_clock')
        memclk_s = f"{memclk}MHz" if memclk is not None else "   -"
        return (f"{c(f'{idx:<3}', 'bold', 'cyan')} {name:<20} "
                f"{c(f'{temp_s:>5}', temp_col)} {fan_s:>4} {util_s:>5} "
                f"{core_s:>8} {memclk_s:>8} {power_s:>11} {mem_s:>13}")

    # ------------------------------------------------------------------
    # Menu definition and rendering
    # ------------------------------------------------------------------
    def _menu_sections(self):
        """Ordered menu: list of (section_title, [(label, handler), ...])."""
        return [
            ("Information & Monitoring", [
                ("Live dashboard (auto-refresh)", self.live_dashboard),
                ("GPU dashboard (full nvidia-smi)", self.action_show_info),
                ("Detailed query (per GPU)", self.action_query_details),
                ("Live monitor — clocks/power (dmon)", self.monitor_gpu_live),
                ("Live monitor — processes (pmon)", self.monitor_processes),
                ("GPU capabilities table", self.show_gpu_capabilities),
                ("Supported clocks (raw dump)", self.action_query_supported_clocks),
            ]),
            ("Clocks & Power", [
                ("Set power limit (validated range)", self.action_set_power_limit),
                ("Set core + memory clocks", self.action_set_app_clocks),
                ("Reset clocks to default", self.action_reset_app_clocks),
                ("Persistence mode on/off", self.action_set_persistence),
                ("GPU operation mode (GOM)", self.action_set_gom),
                ("Auto-boost on/off", self.action_set_auto_boost),
            ]),
            ("Memory & Compute", [
                ("Compute mode", self.action_set_compute_mode),
                ("ECC memory on/off", self.action_toggle_ecc),
                ("MIG mode on/off", self.action_toggle_mig),
                ("Accounting mode on/off", self.action_toggle_accounting),
                ("Clear accounting data", self.action_clear_accounting),
            ]),
            ("Fan & Hardware", [
                ("Set GPU fan speed", self.action_set_fan_speed),
                ("Sensors & fan control (lm-sensors)", self.fan_control_menu),
                ("Reset GPU", self.action_reset_gpu),
            ]),
            ("Profiles & System", [
                ("vLLM optimizer (multi-GPU / tensor-parallel)", self.action_launch_vllm),
                ("Apply all saved settings", self.apply_all_settings),
                ("Show saved settings", self.show_current_settings),
                ("Generate systemd service", self.generate_systemd_service),
                ("Reset ALL GPUs to defaults", self.action_reset_all),
            ]),
        ]

    def _screen_title(self, title: str, subtitle: Optional[str] = None, width: int = 74):
        """Draw a styled boxed title used across screens."""
        top = "╔" + "═" * width + "╗"
        bot = "╚" + "═" * width + "╝"
        print(c(top, 'cyan'))
        print(c("║", 'cyan') + c(title.center(width), 'bold', 'white') + c("║", 'cyan'))
        if subtitle:
            print(c("║", 'cyan') + c(subtitle.center(width), 'gray') + c("║", 'cyan'))
        print(c(bot, 'cyan'))

    def _render_header(self):
        width = 74
        title = "NVIDIA GPU CONTROL PANEL"
        drv = f"Driver {self.driver_version}   CUDA {self.cuda_version}"
        gpu_names = {g.get('name', '') for g in self.gpus}
        if len(gpu_names) == 1 and self.gpu_count:
            summary = f"{self.gpu_count}× {next(iter(gpu_names))}"
        else:
            summary = f"{self.gpu_count} GPU(s)"

        top = "╔" + "═" * width + "╗"
        mid = "╠" + "═" * width + "╣"
        bot = "╚" + "═" * width + "╝"

        def row(content):
            pad = width - Style.strip_len(content)
            left = pad // 2
            return "║" + " " * left + content + " " * (pad - left) + "║"

        print(c(top, 'cyan'))
        print(c("║", 'cyan') + c(title.center(width), 'bold', 'white') + c("║", 'cyan'))
        print(c(mid, 'cyan'))
        info = c(drv, 'gray') + c("   •   ", 'cyan') + c(summary, 'green')
        print(c("║", 'cyan') + " " * ((width - Style.strip_len(info)) // 2) + info +
              " " * (width - Style.strip_len(info) - (width - Style.strip_len(info)) // 2) +
              c("║", 'cyan'))
        print(c(bot, 'cyan'))

    def _render_status(self):
        rows = self.get_live_status()
        if not rows:
            return
        header = f"  {'GPU':<3} {'NAME':<26} {'TEMP':>5} {'UTIL':>5} {'POWER':>11} {'MEMORY':>13}"
        print(c(header, 'bold', 'gray'))
        for r in rows:
            print("  " + self._format_status_row(r, wide=False))

    def display_menu(self):
        """Display the main menu (data-driven, styled)."""
        while True:
            os.system('clear')
            self._render_header()
            print()
            self._render_status()
            print()

            # Build the numbered command map while rendering the sections.
            sections = self._menu_sections()
            command_map = {}
            n = 1
            for title, items in sections:
                print(c(f"  {title.upper()}", 'bold', 'cyan'))
                for label, handler in items:
                    command_map[str(n)] = handler
                    num = c(f"[{n:>2}]", 'yellow')
                    print(f"   {num}  {label}")
                    n += 1
                print()

            print(c("   [ 0]", 'red') + c("  Exit", 'red'))
            print(c("  " + "─" * 74, 'gray'))

            try:
                choice = input(c("\n  Select › ", 'bold', 'green')).strip()
                if choice == '0':
                    break
                handler = command_map.get(choice)
                if handler:
                    handler()
                else:
                    print(c("  Invalid choice", 'red'))
                    time.sleep(0.8)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(c(f"  Error: {e}", 'red'))
                self._pause()
    
    def generate_systemd_service(self):
        """Generate systemd service for boot persistence"""
        service_content = f"""[Unit]
Description=NVIDIA GPU Settings Persistence
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 {os.path.abspath(__file__)} --apply-settings
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""
        service_file = Path('/etc/systemd/system/nvidia-settings-persistence.service')
        
        print("\nSystemd service content:")
        print("=" * 60)
        print(service_content)
        print("=" * 60)
        print(f"\nTo install the service, run:")
        print(f"1. sudo tee {service_file} << EOF")
        print(service_content)
        print("EOF")
        print(f"2. sudo systemctl daemon-reload")
        print(f"3. sudo systemctl enable nvidia-settings-persistence.service")
        print(f"4. sudo systemctl start nvidia-settings-persistence.service")
        
        if input("\nAutomatically install service? (requires sudo) (y/n): ").lower() == 'y':
            try:
                # Write service file
                subprocess.run(['sudo', 'tee', str(service_file)], 
                             input=service_content.encode(), 
                             capture_output=True)
                # Reload systemd
                subprocess.run(['sudo', 'systemctl', 'daemon-reload'], capture_output=True)
                # Enable service
                subprocess.run(['sudo', 'systemctl', 'enable', 'nvidia-settings-persistence.service'], 
                             capture_output=True)
                print("✓ Service installed and enabled successfully")
            except Exception as e:
                print(f"✗ Failed to install service: {e}")
        
        input("\nPress Enter to continue...")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--apply-settings':
        # Apply settings mode (for systemd service)
        controller = NvidiaGPUController()
        controller.apply_all_settings()
    else:
        # Interactive menu mode
        controller = NvidiaGPUController()
        if controller.gpu_count == 0:
            print("No NVIDIA GPUs detected!")
            print("Make sure nvidia-smi is installed and working.")
            sys.exit(1)
        
        controller.display_menu()
        print("\nGoodbye!")

if __name__ == "__main__":
    main()