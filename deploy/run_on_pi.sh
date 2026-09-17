#!/usr/bin/env bash
# One-shot Raspberry Pi 5 test of an exported model, over SSH, from this machine.
#
#   deploy/run_on_pi.sh pi@raspberrypi.local              # the MVP: tcn_mla_eog (TCN + 2-layer MLA)
#   deploy/run_on_pi.sh pi@192.168.1.42 tcn_eog           # any other exported model
#   HOP=64 THREADS=3 deploy/run_on_pi.sh pi@raspberrypi.local
#
# What it does
#   1. checks the five files the Pi needs exist here (exports are not in git;
#      run `python -m deploy.export_onnx --ckpt runs/<model>/best.pt` first)
#   2. copies them to ~/$REMOTE_DIR on the Pi
#   3. on the Pi: creates a venv on first run, installs numpy + onnxruntime,
#      prints the board / Python / onnxruntime versions, then runs
#      raspi_infer.py --benchmark --testdata, recording SoC temperature,
#      ARM clock and the throttle flag before/after (peak temp during)
# Re-running is cheap: the venv is reused and only the copy + test repeat.
#
# Needs: ssh access to the Pi (key or password -- asked once thanks to connection
# sharing), python3 with venv on the Pi (Raspberry Pi OS ships it; otherwise
# `sudo apt install python3-venv`).  If the Pi is plugged straight into this
# Mac with Internet Sharing, its address is usually 192.168.2.2.
set -euo pipefail

PI="${1:?usage: $0 user@host [model_name]   e.g. $0 pi@raspberrypi.local tcn_mla_eog}"
MODEL="${2:-tcn_mla_eog}"
REMOTE_DIR="${REMOTE_DIR:-eeg}"          # relative to the Pi user's home
TESTDATA="${TESTDATA:-EOG_test100.npz}"  # file in deploy/testdata/
HOP="${HOP:-32}"
THREADS="${THREADS:-2}"                  # 2 measured best on a Pi 5 (4 gives a 2x worse p95)

DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILES=(
  "$DEPLOY/raspi_infer.py"
  "$DEPLOY/requirements-raspi.txt"
  "$DEPLOY/artifacts/$MODEL.onnx"
  "$DEPLOY/artifacts/$MODEL.json"
  "$DEPLOY/testdata/$TESTDATA"
)
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "missing: $f" >&2
    case "$f" in
      *.onnx|*.json) echo "export it first:  python -m deploy.export_onnx --ckpt runs/$MODEL/best.pt" >&2 ;;
      *.npz)         echo "cut it first:     python -m deploy.make_testdata --artifact EOG --n 100" >&2 ;;
    esac
    exit 1
  fi
done

# Share one SSH connection across the three hops so a password is asked at most once.
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=/tmp/run_on_pi-%C" -o ControlPersist=120 -o StrictHostKeyChecking=accept-new)
trap 'ssh "${SSH_OPTS[@]}" -O exit "$PI" 2>/dev/null || true' EXIT

echo "==> copying ${#FILES[@]} files to $PI:~/$REMOTE_DIR"
ssh "${SSH_OPTS[@]}" "$PI" "mkdir -p ~/$REMOTE_DIR"
scp -q "${SSH_OPTS[@]}" "${FILES[@]}" "$PI:~/$REMOTE_DIR/"

echo "==> running on $PI"
ssh "${SSH_OPTS[@]}" "$PI" "REMOTE_DIR='$REMOTE_DIR' MODEL='$MODEL' TESTDATA='$TESTDATA' HOP='$HOP' THREADS='$THREADS' bash -s" <<'REMOTE'
set -euo pipefail
cd ~/"$REMOTE_DIR"
if [[ ! -x env/bin/python ]]; then
  echo "--> creating venv (first run only)"
  python3 -m venv env || { echo "python3 -m venv failed; try: sudo apt install python3-venv" >&2; exit 1; }
fi
env/bin/python -m pip install -q --disable-pip-version-check -r requirements-raspi.txt
BOARD="$( { tr -d '\0' < /proc/device-tree/model; } 2>/dev/null || uname -m)"
echo "--> $BOARD | $(env/bin/python --version) | onnxruntime $(env/bin/python -c 'import onnxruntime as o; print(o.__version__)') | $(env/bin/python -c 'import os; print(os.cpu_count())') cores"

# --- thermal / clock telemetry (Raspberry Pi OS; degrades to n/a elsewhere) ---
temp()  { if command -v vcgencmd >/dev/null 2>&1; then vcgencmd measure_temp | tr -dc '0-9.';
          elif [[ -r /sys/class/thermal/thermal_zone0/temp ]]; then awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp;
          else echo n/a; fi; }
clock() { vcgencmd measure_clock arm 2>/dev/null | awk -F= '{printf "%d MHz", $2/1e6}' || true; }
thr()   { vcgencmd get_throttled 2>/dev/null | cut -d= -f2 || true; }   # 0x0 = never throttled since boot
PEAK_FILE="$(mktemp)"; echo "$(temp)" > "$PEAK_FILE"
( while :; do t="$(temp)"; p="$(cat "$PEAK_FILE")"
    [[ "$t" != n/a ]] && awk -v a="$t" -v b="$p" 'BEGIN{exit !(a>b)}' && echo "$t" > "$PEAK_FILE"; sleep 2; done ) &
SAMPLER=$!
echo "--> before: $(temp)°C  $(clock)  throttled=$(thr)"

env/bin/python raspi_infer.py "$MODEL.onnx" --threads "$THREADS" --hop "$HOP" --benchmark --testdata "$TESTDATA"

kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null || true
echo "--> after:  $(temp)°C  $(clock)  throttled=$(thr)   (peak during test: $(cat "$PEAK_FILE")°C)"
rm -f "$PEAK_FILE"
REMOTE
echo "==> done"
