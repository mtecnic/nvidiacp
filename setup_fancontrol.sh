#!/usr/bin/env bash
#
# setup_fancontrol.sh — set up lm-sensors + fancontrol so chassis PWM fans
# become controllable (exposes /sys/class/hwmon/*/pwmN files).
#
# Run with root:   sudo ./setup_fancontrol.sh
# From Claude Code prompt:   ! sudo bash setup_fancontrol.sh
#
set -uo pipefail

# --- must be root (sudo needs a terminal, so run the whole script as root) ---
if [[ $EUID -ne 0 ]]; then
    echo "This script must run as root. Re-run with:  sudo $0" >&2
    exit 1
fi

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*"; }

# --- 1. install packages -----------------------------------------------------
log "Installing lm-sensors and fancontrol"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y lm-sensors fancontrol
ok "packages installed"

# --- 2. safe auto-probe ------------------------------------------------------
log "Running safe sensor auto-probe (sensors-detect --auto)"
# --auto only performs low-risk scans and appends found modules to /etc/modules.
sensors-detect --auto || warn "sensors-detect returned non-zero (often harmless)"

# --- 3. load any modules it recommends --------------------------------------
log "Loading detected sensor modules"
MODFILE=""
for f in /etc/modules /etc/modules-load.d/lm-sensors.conf; do
    [[ -f "$f" ]] && MODFILE="$f $MODFILE"
done

loaded_any=0
# Common motherboard Super I/O drivers that expose PWM/fan control.
CANDIDATES="nct6775 nct6683 it87 w83627ehf w83627hf f71882fg f71805f smsc47m1 dme1737 asus_wmi_sensors asus_ec_sensors"

# First, try whatever sensors-detect wrote into the module files.
for f in $MODFILE; do
    while read -r mod _; do
        [[ -z "$mod" || "$mod" == \#* ]] && continue
        if modprobe "$mod" 2>/dev/null; then
            ok "loaded $mod"
            loaded_any=1
        fi
    done < "$f"
done

# Fall back to probing common Super I/O drivers directly.
if [[ $loaded_any -eq 0 ]]; then
    warn "no modules from sensors-detect; trying common Super I/O drivers"
    for mod in $CANDIDATES; do
        if modprobe "$mod" 2>/dev/null; then
            ok "loaded $mod"
            loaded_any=1
        fi
    done
fi

# --- 4. report ---------------------------------------------------------------
log "Current sensor readings"
sensors || warn "sensors produced no output"

log "PWM fan channels now exposed under /sys/class/hwmon"
found=0
for hwmon in /sys/class/hwmon/hwmon*; do
    name=$(cat "$hwmon/name" 2>/dev/null)
    for pwm in "$hwmon"/pwm[0-9]*; do
        base=$(basename "$pwm")
        [[ "$base" =~ ^pwm[0-9]+$ ]] || continue   # skip pwmN_enable etc.
        val=$(cat "$pwm" 2>/dev/null)
        printf '  %s: %s = %s\n' "$name" "$base" "${val:-?}"
        found=1
    done
done

echo
if [[ $found -eq 1 ]]; then
    ok "PWM channels detected — fan control is available."
    echo "Next steps:"
    echo "  • Build automatic fan curves:   sudo pwmconfig   (then: systemctl enable --now fancontrol)"
    echo "  • Or drive fans manually from the panel:  python3 nvidia_control.py  → option 24"
else
    warn "No PWM channels found."
    echo "Your motherboard's Super I/O chip may not be supported by a Linux driver,"
    echo "or a BIOS/board-specific driver is needed. Review the 'sensors-detect' output"
    echo "above for a suggested chip/module."
fi
