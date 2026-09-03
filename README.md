# NVIDIA Control Panel (nvidiacp)

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NVIDIA](https://img.shields.io/badge/GPU-NVIDIA-orange.svg)](https://www.nvidia.com)

A full-featured text menu for controlling NVIDIA GPUs on Linux — power, clocks,
fans, ECC, MIG, compute mode — with **boot persistence**, a **closed-loop fan
controller**, and a built-in **vLLM launch optimizer** for LLM serving.

Everything runs through `nvidia-smi` + NVML (vendored, zero pip installs).
Headless-friendly: no X server, no coolbits required.

## Features

- **Complete GPU control** — live dashboard, dmon/pmon monitors, power limits,
  app clocks, persistence mode, GOM, auto-boost, compute mode, ECC, MIG,
  accounting
- **Fan control without X** — fixed % or automatic via NVML (root), clamped to
  each card's real min/max
- **Temperature-curve fan daemon** — closed-loop controller that holds target
  temps, rides a tuned curve, and falls back to VBIOS idle band when cold
- **vLLM optimizer** — multi-GPU / tensor-parallel launch plans with 5
  workload profiles, sized to your actual VRAM
- **Boot persistence** — settings survive reboots via systemd
- **Multi-GPU** — per-GPU or all-GPU operations, mixed-fleet aware
- **Zero dependencies** — pure Python 3 standard library + vendored `pynvml.py`

## Screenshot

```
╔══════════════════════════════════════════════════════════════════════╗
║                  NVIDIA GPU CONTROL PANEL                           ║
╠══════════════════════════════════════════════════════════════════════╣
║        Driver 570.xx   •   CUDA 12.8   •   4× NVIDIA RTX 3090        ║
╚══════════════════════════════════════════════════════════════════════╝

  GPU  NAME                       TEMP   UTIL      POWER        MEMORY
  ─── ────────────────────────── ───── ───── ───────────── ─────────────
  [ 0] RTX 3090                   47°C   12%   42.3W/350.0W    5.1/24GB
  [ 1] RTX 3090                   45°C    3%   28.1W/350.0W    1.2/24GB

  INFORMATION & MONITORING
   [ 1]  Live dashboard (auto-refresh)
   [ 2]  GPU dashboard (full nvidia-smi)
   [ 3]  Detailed query (per GPU)
   [ 4]  Live monitor — clocks/power (dmon)
   [ 5]  Live monitor — processes (pmon)
   [ 6]  GPU capabilities table
   [ 7]  Supported clocks (raw dump)

  CLOCKS & POWER
   [ 8]  Set power limit (validated range)
   [ 9]  Set core + memory clocks
  [ 10]  Reset clocks to default
  [ 11]  Persistence mode on/off
  [ 12]  GPU operation mode (GOM)
  [ 13]  Auto-boost on/off

  MEMORY & COMPUTE
  [ 14]  Compute mode
  [ 15]  ECC memory on/off
  [ 16]  MIG mode on/off
  [ 17]  Accounting mode on/off
  [ 18]  Clear accounting data

  FAN & HARDWARE
  [ 19]  Set GPU fan speed (fixed %)
  [ 20]  Auto fan (temperature curve)
  [ 21]  Reset GPU fan to auto
  [ 22]  Sensors & fan control (lm-sensors)
  [ 23]  Reset GPU

  PROFILES & SYSTEM
  [ 24]  vLLM optimizer (multi-GPU / tensor-parallel)
  [ 25]  Apply all saved settings
  [ 26]  Show saved settings
  [ 27]  Generate systemd service
  [ 28]  Reset ALL GPUs to defaults

   [ 0]  Exit

  Select ›
```

## Installation

```bash
git clone https://github.com/mtecnic/nvidiacp.git
cd nvidiacp
sudo ./install.sh
```

The installer creates the `nvidiacp` command and enables the boot-persistence
systemd service. See [INSTALL.md](INSTALL.md) for details.

Run without installing:

```bash
sudo python3 nvidia_control.py
```

## Fan control

### Fixed speed

Menu option **Set GPU fan speed** drives every fan on the card directly via
NVML as root — no X server or coolbits. Speeds are clamped to the card's own
min/max (queried per card) and persisted to `/etc/nvidiacp`.

### Automatic (temperature curve)

Menu option **Auto fan (temperature curve)** marks a GPU for the closed-loop
`nvidiacp-fan.service` daemon:

- **Curve**: 60°C→40%, 68°C→70%, 75°C→85%, 80°C→95%, 83°C→100%
  (linear interpolation between points)
- **Hard safety**: ≥85°C forces 100% (core throttle starts ~83°C)
- **Idle band**: ≤55°C (with 3°C hysteresis) hands the card back to its VBIOS
  idle curve — quiet and stable when cold
- **Anti-hunt**: ramp-up follows the curve immediately; spin-down is limited
  to 3%/tick
- **Live targets**: the daemon re-reads the settings file every 3s tick, so
  TUI changes apply without a restart
- **Clean exit**: SIGTERM/SIGINT returns every managed GPU to automatic
  (VBIOS) control

Check the daemon: `sudo systemctl status nvidiacp-fan` ·
logs: `journalctl -u nvidiacp-fan -e`

## vLLM optimizer

Menu option **vLLM optimizer** picks GPUs and a workload profile, applies the
matching power/clock/persistence settings, and prints a ready-to-paste launch
plan sized to your **minimum** per-GPU VRAM (mixed fleets warned):

| Profile | Power | Clocks | Mem util | Extra flags |
|---|---|---|---|---|
| **Maximum Throughput** | 100% | locked max | 0.95 | chunked-prefill, prefix-caching |
| **Low Latency** | 100% | locked max | 0.90 | enforce-eager |
| **Balanced** | 90% | default | 0.90 | — |
| **Power Efficient** | 70% | default (boost) | 0.85 | — |
| **Production Ready** | 90% | default | 0.90 | persistence on |

Example output (2× RTX 3090, max throughput):

```
── Suggested vLLM launch ────────────────────────────────────────
  2-way tensor parallel · ~48GB total (24GB/GPU)

  CUDA_VISIBLE_DEVICES=0,1 \
  vllm serve <MODEL> \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --max-num-seqs 96 \
    --enable-chunked-prefill \
    --enable-prefix-caching

  Env: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Context/batch heuristics scale with GPU family (A100/H100 → 32K/256,
A6000/RTX 6000 → 16K/128, 4090/4080 → 8K/64, 3090 → 8K/48, fallback by VRAM).
These are sensible starting points — tune to your model.

## Persistence

Settings live in `/etc/nvidiacp/settings.json` (shared with the boot service)
and per-user at `~/.config/nvidiacp/settings.json`.

```bash
# applied at boot by nvidia-settings-persistence.service
python3 nvidia_control.py --apply-settings

# daemon modes (used by systemd units)
python3 nvidia_control.py --fan-daemon
python3 nvidia_control.py --nvml-fan <set|auto> <gpu> [pct]
```

## Requirements

- NVIDIA GPU with drivers + `nvidia-smi`
- Python 3.6+ (standard library only)
- Root for writes (sudo)
- `lm-sensors` optional — chassis/motherboard PWM fans

## Uninstall

```bash
sudo ./uninstall.sh
```

## License

MIT — see [LICENSE](LICENSE).
