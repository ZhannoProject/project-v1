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
#      raspi_infer.py --benchmark --testdata --latency, recording SoC temperature,
#      ARM clock and the throttle flag before/after (peak temp during)
#   4. optionally (REALTIME_S=30) replays a 30 s continuous signal cut from the
#      test set at the sample rate through StreamingDenoiser -- the live-loop
#      test: paced per-hop cost, end-to-end latency, interval jitter
#   5. optionally (DUTY_HOPS=64,32,16,8,4) runs the duty-cycle check: a paced
#      replay per hop size with compute / period, e2e latency, drift and an
#      OK / MARGINAL / FAIL verdict -- is real-time processing applicable, and
#      down to which hop
#   6. copies the per-hop latency traces back to deploy/artifacts/pi_latency/
#      (<model>_hop<hop>[_realtime|_duty].json + .csv)
# Re-running is cheap: the venv is reused and only the copy + test repeat.
#   LATENCY_EVERY=200 prints a rolling latency line every N hops (0 = summary only).
#   REALTIME_S=30 adds the paced replay (takes that many seconds; 0 = skip).
#   DUTY_HOPS=64,32,16,8,4 DUTY_S=20 adds the duty-cycle sweep (DUTY_S seconds per hop).
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
LATENCY_EVERY="${LATENCY_EVERY:-400}"    # rolling per-hop latency line every N hops (1600 hops in the bundled set)
LATENCY_STEM="latency_${MODEL}_hop${HOP}" # per-hop trace written on the Pi, copied back below
REALTIME_S="${REALTIME_S:-0}"            # seconds of paced live-loop replay (0 = skip)
DUTY_HOPS="${DUTY_HOPS:-}"               # e.g. 64,32,16,8,4 -> duty-cycle sweep (empty = skip)
DUTY_S="${DUTY_S:-20}"                   # paced seconds per hop in the sweep

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
ssh "${SSH_OPTS[@]}" "$PI" "REMOTE_DIR='$REMOTE_DIR' MODEL='$MODEL' TESTDATA='$TESTDATA' HOP='$HOP' THREADS='$THREADS' LATENCY_EVERY='$LATENCY_EVERY' LATENCY_STEM='$LATENCY_STEM' REALTIME_S='$REALTIME_S' DUTY_HOPS='$DUTY_HOPS' DUTY_S='$DUTY_S' bash -s" <<'REMOTE'
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

env/bin/python raspi_infer.py "$MODEL.onnx" --threads "$THREADS" --hop "$HOP" --benchmark --testdata "$TESTDATA" \
    --latency --latency_every "$LATENCY_EVERY" --latency_out "$LATENCY_STEM"

if [[ "$REALTIME_S" -gt 0 ]]; then
  # A continuous recording for the live-loop test: the first REALTIME_S seconds
  # of the test epochs laid end to end (content is irrelevant for timing).
  env/bin/python - "$TESTDATA" "$REALTIME_S" <<'PY'
import sys, numpy as np
d = np.load(sys.argv[1]); fs = int(d["fs"]); n = int(float(sys.argv[2]) * fs)
sig = d["noisy"].reshape(-1)[:n].astype(np.float32)
np.save("realtime_rec.npy", sig)
print(f"--> realtime replay: {len(sig)} samples = {len(sig) / fs:.1f} s at {fs} Hz")
PY
  echo "--> $(temp)°C  $(clock) before paced replay"
  env/bin/python raspi_infer.py "$MODEL.onnx" --threads "$THREADS" --hop "$HOP" \
      --input realtime_rec.npy --stream --realtime \
      --latency --latency_every "$(( LATENCY_EVERY > 0 ? LATENCY_EVERY / 8 : 0 ))" --latency_out "${LATENCY_STEM}_realtime"
fi

if [[ -n "$DUTY_HOPS" ]]; then
  echo "--> $(temp)°C  $(clock) before duty-cycle sweep"
  env/bin/python raspi_infer.py "$MODEL.onnx" --threads "$THREADS" --testdata "$TESTDATA" \
      --duty_cycle "$DUTY_HOPS" --duty_seconds "$DUTY_S" --latency_out "$LATENCY_STEM"
fi

kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null || true
echo "--> after:  $(temp)°C  $(clock)  throttled=$(thr)   (peak during test: $(cat "$PEAK_FILE")°C)"
rm -f "$PEAK_FILE"
REMOTE

LAT_DIR="$DEPLOY/artifacts/pi_latency"
mkdir -p "$LAT_DIR"
scp -q "${SSH_OPTS[@]}" "$PI:~/$REMOTE_DIR/${LATENCY_STEM}*.json" "$PI:~/$REMOTE_DIR/${LATENCY_STEM}*.csv" "$LAT_DIR/"
echo "==> per-hop latency traces:"; ls -1 "$LAT_DIR/${LATENCY_STEM}"*
echo "==> done"
