#!/usr/bin/env bash
# One-shot Raspberry Pi 5 test of an exported model, over SSH, from this machine.
#
#   deploy/run_on_pi.sh pi@raspberrypi.local              # the MVP: tcn_mla_eog (TCN + 2-layer MLA)
#   deploy/run_on_pi.sh pi@192.168.1.42 tcn_eog           # any other exported model
#   HOP=64 THREADS=3 deploy/run_on_pi.sh pi@raspberrypi.local
#
# What it does
#   1. checks the six files the Pi needs exist here (exports are not in git;
#      run `python -m deploy.export_onnx --ckpt runs/<model>/best.pt` first)
#   2. copies them to ~/$REMOTE_DIR on the Pi
#   3. on the Pi: creates a venv on first run, installs numpy + onnxruntime,
#      prints the board / Python / onnxruntime versions, optionally sets the CPU
#      governor (GOVERNOR=performance, needs passwordless sudo), then runs
#      raspi_infer.py --benchmark --testdata --latency, recording SoC temperature,
#      ARM clock and the throttle flag before/after (peak temp during).
#      Every run first pins to $CPUS and records the state of the machine; with
#      REQUIRE_LOCKED=1 it refuses to measure a board that is not held still.
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
#   CPUS=2-3 pins the process to those cores. The default is the top THREADS cores
#     of the board, which keeps the measurement off CPU 0 (device interrupts and
#     the timer tick land there). CPUS= (empty) measures without pinning.
#   GOVERNOR=performance sets the CPU governor first (passwordless sudo on the Pi;
#     the previous governor is reported and NOT restored -- set it back yourself).
#   REQUIRE_LOCKED=1 refuses to measure unless the board is held still.
#   SPIN=0 stops onnxruntime's thread pool busy-waiting between inferences (less
#     CPU left burning between hops, about a millisecond more latency each).
#   CHECK_ONLY=1 reports the board's state and stops, measuring nothing.
#   SSH_CONNECT_TIMEOUT=20 SSH_CONNECT_ATTEMPTS=3 tune how long to wait for the Pi
#     to answer. The timeout covers the TCP connect AND the SSH key exchange, so
#     a board on power-saving WiFi may need more than the default.
#
# Needs: ssh access to the Pi (key or password -- asked once thanks to connection
# sharing), python3 with venv on the Pi (Raspberry Pi OS ships it; otherwise
# `sudo apt install python3-venv`).  If the Pi is plugged straight into this
# Mac with Internet Sharing, its address is usually 192.168.2.2.
set -euo pipefail

PI="${1:?usage: $0 user@host [model_name]   e.g. $0 pi@raspberrypi.local tcn_mla_eog}"
MODEL="${2:-tcn_mla_eog}"
# INCREMENTAL=1 runs <MODEL>_step.onnx -- the one-sample graph that carries its
# state in and out -- instead of the windowed one. Same runtime on the board
# (onnxruntime + numpy); what changes is that the KV cache exists there at all.
INCREMENTAL="${INCREMENTAL:-}"
if [[ -n "$INCREMENTAL" ]]; then
  GRAPH="${MODEL}_step"
else
  GRAPH="$MODEL"
fi
REMOTE_DIR="${REMOTE_DIR:-eeg}"          # relative to the Pi user's home
TESTDATA="${TESTDATA:-EOG_test100.npz}"  # file in deploy/testdata/
HOP="${HOP:-32}"
THREADS="${THREADS:-2}"                  # 2 measured best on a Pi 5 (4 gives a 2x worse p95)
LATENCY_EVERY="${LATENCY_EVERY:-400}"    # rolling per-hop latency line every N hops (1600 hops in the bundled set)
# The trace name has to separate the two paths as well as the model: a step
# run emits per sample, so it is hop 1 whatever HOP says, and naming it after
# HOP would have it overwrite the windowed measurement of the same model --
# two different numbers for two different methods, in one file.
if [[ -n "$INCREMENTAL" ]]; then
  LATENCY_STEM="incremental_${MODEL}_hop1"
else
  LATENCY_STEM="latency_${MODEL}_hop${HOP}"
fi
REALTIME_S="${REALTIME_S:-0}"            # seconds of paced live-loop replay (0 = skip)
DUTY_HOPS="${DUTY_HOPS:-}"               # e.g. 64,32,16,8,4 -> duty-cycle sweep (empty = skip)
DUTY_S="${DUTY_S:-20}"                   # paced seconds per hop in the sweep
# --- the locked state the measurement is taken under ---
# Cores to pin to.  "auto" is resolved on the Pi, where the core count is known,
# to the top THREADS of them -- counting down from the end keeps the measurement
# off CPU 0, which takes the device interrupts and the timer tick on a stock
# Raspberry Pi OS.  An explicitly empty value means "do not pin".
CPUS="${CPUS-auto}"
GOVERNOR="${GOVERNOR:-}"                 # e.g. performance (needs passwordless sudo; empty = leave it)
REQUIRE_LOCKED="${REQUIRE_LOCKED:-}"     # 1 = refuse to measure a board that is not held still
CHECK_ONLY="${CHECK_ONLY:-}"             # 1 = report the board's state and stop (no measurement)
SPIN="${SPIN:-1}"                        # 0 = stop onnxruntime's thread pool busy-waiting between hops

DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILES=(
  "$DEPLOY/raspi_infer.py"
  "$DEPLOY/preflight.py"
  "$DEPLOY/requirements-raspi.txt"
  "$DEPLOY/artifacts/$GRAPH.onnx"
  "$DEPLOY/artifacts/$GRAPH.json"
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
# ConnectTimeout covers the TCP connect *and* the SSH handshake and key
# exchange, not just the connect -- so it has to fit a Pi 5 waking its WiFi
# radio (power saving is on by default in Raspberry Pi OS) plus a full KEX.
# ConnectionAttempts is 1 unless it is said otherwise, which turns one dropped
# mDNS or ARP packet into a failed run; .local names are multicast, sent at the
# lowest WiFi rate, and dropped often. Three tries at 20 s is roughly the OS
# default TCP timeout in total, and survives a single lost packet.
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-20}"
SSH_CONNECT_ATTEMPTS="${SSH_CONNECT_ATTEMPTS:-3}"
SSH_OPTS+=(-o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" -o "ConnectionAttempts=$SSH_CONNECT_ATTEMPTS"
           # ConnectTimeout only governs setting the connection up. A sweep runs
           # for minutes, and a link that dies mid-measurement would otherwise
           # hang until TCP gave up -- which, for an idle connection behind NAT,
           # can be never. Give up after ~2 min of an unanswering link instead.
           -o ServerAliveInterval=15 -o ServerAliveCountMax=8)
# With no terminal -- run from the GUI's Deploy tab, or from any script -- there
# is nobody to answer a password prompt or a host-key question, so ask ssh to
# fail instead of waiting for an answer that cannot come. Key-based auth is then
# required, which is what an unattended measurement needs anyway.
if [[ ! -t 0 ]]; then
  SSH_OPTS+=(-o BatchMode=yes)
fi
trap 'ssh "${SSH_OPTS[@]}" -O exit "$PI" 2>/dev/null || true' EXIT

# scp splits user@host:path on the first colon, so an IPv6 literal has to be
# bracketed -- "changlong@fe80::...%en5" makes scp try to resolve the host
# "fe80".  ssh takes it either way, so only the scp target is rewritten.  This
# is how a Pi on a direct Ethernet link is usually reachable: mDNS gives it a
# link-local IPv6 address and no IPv4 at all.
case "${PI##*@}" in
  *:*) if [[ "$PI" == *@* ]]; then SCP_PI="${PI%@*}@[${PI##*@}]"; else SCP_PI="[$PI]"; fi ;;
  *)   SCP_PI="$PI" ;;
esac

# One cheap connection before the 1.2 MB copy, so a failure is diagnosed rather
# than guessed at. Every one of these looks the same from the GUI -- "it timed
# out" -- and they need completely different things done about them.
echo "==> connecting to $PI  (${SSH_CONNECT_TIMEOUT}s timeout, ${SSH_CONNECT_ATTEMPTS} attempts)"
if ! PROBE="$(ssh "${SSH_OPTS[@]}" "$PI" true 2>&1)"; then
  [[ -n "$PROBE" ]] && echo "$PROBE" >&2
  HOST="${PI##*@}"
  case "$PROBE" in
    *"Could not resolve"*|*"Name or service not known"*|*"nodename nor servname"*)
      echo "--> '$HOST' did not resolve. A .local name is answered over mDNS, which WiFi" >&2
      echo "    drops routinely; the Pi also has to be awake to answer it. Use its address" >&2
      echo "    instead -- 'hostname -I' on the Pi -- e.g. $0 ${PI%%@*}@192.168.1.42" >&2 ;;
    *"Permission denied"*)
      echo "--> no password can be typed here (there is no terminal), so the Pi needs your" >&2
      echo "    key. Run this once from a terminal:  ssh-copy-id $PI" >&2 ;;
    *"Host key verification failed"*|*"IDENTIFICATION HAS CHANGED"*)
      echo "--> the Pi's host key is not the one on record (reflashed, or a reused address)." >&2
      echo "    If that is expected:  ssh-keygen -R $HOST" >&2 ;;
    *"timed out"*|*"Timeout"*|*"No route to host"*|*"Connection refused"*|"")
      echo "--> no usable answer in ${SSH_CONNECT_TIMEOUT}s x ${SSH_CONNECT_ATTEMPTS}. That budget covers the TCP connect" >&2
      echo "    *and* the SSH key exchange, so a Pi on power-saving WiFi can legitimately" >&2
      echo "    need longer:  SSH_CONNECT_TIMEOUT=45 $0 $PI" >&2
      echo "    Check it is up and sshd is running:  ping $HOST" >&2 ;;
  esac
  exit 1
fi

echo "==> copying ${#FILES[@]} files to $PI:~/$REMOTE_DIR"
ssh "${SSH_OPTS[@]}" "$PI" "mkdir -p ~/$REMOTE_DIR"
scp -q "${SSH_OPTS[@]}" "${FILES[@]}" "$SCP_PI:~/$REMOTE_DIR/"

# Everything the remote half reads, named once. The preamble is built from this
# list rather than spelled out again in the ssh command, because the two had to
# be kept in step by hand and were not: SPIN reached the Pi unset and `set -u`
# killed the run at line 45 -- after the files had been copied and the governor
# had been changed. %q quoting also survives a value with a space or a quote in
# it, which the hand-written 'single quotes' did not.
REMOTE_VARS=(REMOTE_DIR MODEL GRAPH INCREMENTAL TESTDATA HOP THREADS LATENCY_EVERY
             LATENCY_STEM REALTIME_S DUTY_HOPS DUTY_S CPUS GOVERNOR REQUIRE_LOCKED
             CHECK_ONLY SPIN)
REMOTE_ENV=""
for v in "${REMOTE_VARS[@]}"; do
  printf -v q '%q' "${!v-}"
  REMOTE_ENV+="$v=$q "
done

echo "==> running on $PI"
ssh "${SSH_OPTS[@]}" "$PI" "$REMOTE_ENV bash -s" <<'REMOTE'
set -euo pipefail
# An array cannot ride through the scalar env above, so it is rebuilt here
# from the flag that did travel.
if [[ -n "$INCREMENTAL" ]]; then STEP_ARGS=(--incremental); else STEP_ARGS=(); fi
cd ~/"$REMOTE_DIR"
if [[ ! -x env/bin/python ]]; then
  echo "--> creating venv (first run only)"
  python3 -m venv env || { echo "python3 -m venv failed; try: sudo apt install python3-venv" >&2; exit 1; }
fi
if ! env/bin/python -m pip install -q --disable-pip-version-check -r requirements-raspi.txt; then
  echo "--> could not install numpy + onnxruntime on the Pi. It needs network access the" >&2
  echo "    first time; after that the venv is reused and nothing is downloaded." >&2
  exit 1
fi
BOARD="$( { tr -d '\0' < /proc/device-tree/model; } 2>/dev/null || uname -m)"
echo "--> $BOARD | $(env/bin/python --version) | onnxruntime $(env/bin/python -c 'import onnxruntime as o; print(o.__version__)') | $(env/bin/python -c 'import os; print(os.cpu_count())') cores"

# --- thermal / clock telemetry (Raspberry Pi OS; degrades to n/a elsewhere) ---
temp()  { if command -v vcgencmd >/dev/null 2>&1; then vcgencmd measure_temp | tr -dc '0-9.';
          elif [[ -r /sys/class/thermal/thermal_zone0/temp ]]; then awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp;
          else echo n/a; fi; }
clock() { vcgencmd measure_clock arm 2>/dev/null | awk -F= '{printf "%d MHz", $2/1e6}' || true; }
thr()   { vcgencmd get_throttled 2>/dev/null | cut -d= -f2 || true; }   # 0x0 = never throttled since boot
# --- lock the clock, if asked and allowed ---
# The governor is the one control that needs root, so it is opt-in and says
# plainly what it managed to do: a measurement taken under "ondemand" while the
# report claims "performance" is worse than one that admits it.
if [[ -n "$GOVERNOR" ]]; then
  WAS="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  if [[ "$WAS" == "$GOVERNOR" ]]; then
    echo "--> CPU governor already $GOVERNOR"
  elif sudo -n true 2>/dev/null; then
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      echo "$GOVERNOR" | sudo -n tee "$g" >/dev/null || true
    done
    NOW="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
    echo "--> CPU governor $WAS -> $NOW  (not restored; put it back with: sudo cpupower frequency-set -g $WAS)"
  else
    echo "--> CPU governor is $WAS and passwordless sudo is not available here;" \
         "set it on the Pi with: sudo cpupower frequency-set -g $GOVERNOR" >&2
  fi
fi
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
# Written out rather than as `[[ ... ]] && ...`, which returns non-zero when the
# condition is false and would end the script under `set -e` if it ever became
# the last line of a block.
# LOCK_ARGS: understood by preflight.py *and* raspi_infer.py.
# RUN_ARGS:   those plus the ones only the measurement takes. Keeping them apart
#             matters -- preflight.py has no --no-spin, so a state check with
#             spinning disabled died on "unrecognized arguments".
if [[ "$CPUS" == auto ]]; then
  # --all, because plain nproc honours OMP_NUM_THREADS -- which is exported
  # just above -- and would report the thread count as the core count, pinning
  # the measurement onto the bottom cores it is meant to stay off.
  NPROC="$(nproc --all 2>/dev/null || nproc 2>/dev/null || echo 4)"
  FIRST=$(( NPROC - THREADS ))
  if (( FIRST < 0 )); then FIRST=0; fi      # more threads than cores: pin to all of them
  CPUS="$FIRST-$(( NPROC - 1 ))"
  echo "--> pinning to CPU $CPUS (the top $THREADS of $NPROC; CPU 0 does the housekeeping)"
fi
LOCK_ARGS=()
if [[ -n "$CPUS" ]]; then LOCK_ARGS+=(--cpus "$CPUS"); fi
if [[ -n "$REQUIRE_LOCKED" ]]; then LOCK_ARGS+=(--require-locked); fi
RUN_ARGS=("${LOCK_ARGS[@]+"${LOCK_ARGS[@]}"}")
# onnxruntime's pool spins before sleeping: it buys about a millisecond per hop
# and costs roughly 1.7x the CPU while the loop is idle between them, so it
# decides how much of the board is left over rather than whether it keeps up.
if [[ "$SPIN" == "0" ]]; then RUN_ARGS+=(--no-spin); fi

# CHECK_ONLY=1: report the state of the board and stop, so the conditions can
# be settled before committing to a sweep that takes minutes.
if [[ -n "${CHECK_ONLY:-}" ]]; then
  env/bin/python preflight.py --threads "$THREADS" "${LOCK_ARGS[@]+"${LOCK_ARGS[@]}"}"
  exit $?
fi

PEAK_FILE="$(mktemp)"; echo "$(temp)" > "$PEAK_FILE"
( while :; do t="$(temp)"; p="$(cat "$PEAK_FILE")"
    [[ "$t" != n/a ]] && awk -v a="$t" -v b="$p" 'BEGIN{exit !(a>b)}' && echo "$t" > "$PEAK_FILE"; sleep 2; done ) &
SAMPLER=$!
# The sampler is a background loop with nothing to stop it: a measurement that
# fails under `set -e`, or a Stop from the GUI, would leave it reading the
# thermal zone every two seconds on the board forever, and leave the temp file
# behind with it. An EXIT trap covers every way out of here but SIGKILL.
trap 'kill "$SAMPLER" 2>/dev/null || true; rm -f "$PEAK_FILE"' EXIT
# Clear this stem's traces on the board before measuring. The copy-back at
# the end is a glob -- ${LATENCY_STEM}*.json -- so anything left here from an
# earlier run comes home with a fresh timestamp and is read as part of *this*
# one. That is how a run with the duty sweep switched off still produced a row
# saying "duty sweep": the file was the previous run's, resurrected by scp.
# Nothing is lost by deleting them; they were copied back when they were made.
rm -f "$LATENCY_STEM".json "$LATENCY_STEM".csv "$LATENCY_STEM"_*.json "$LATENCY_STEM"_*.csv

echo "--> before: $(temp)°C  $(clock)  throttled=$(thr)"

env/bin/python raspi_infer.py "$GRAPH.onnx" --threads "$THREADS" --hop "$HOP" \
    "${STEP_ARGS[@]+"${STEP_ARGS[@]}"}" "${RUN_ARGS[@]+"${RUN_ARGS[@]}"}" \
    --benchmark --testdata "$TESTDATA" \
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
  env/bin/python raspi_infer.py "$GRAPH.onnx" --threads "$THREADS" --hop "$HOP" \
    "${STEP_ARGS[@]+"${STEP_ARGS[@]}"}" "${RUN_ARGS[@]+"${RUN_ARGS[@]}"}" \
      --input realtime_rec.npy --stream --realtime \
      --latency --latency_every "$(( LATENCY_EVERY > 0 ? LATENCY_EVERY / 8 : 0 ))" --latency_out "${LATENCY_STEM}_realtime"
fi

if [[ -n "$DUTY_HOPS" ]]; then
  echo "--> $(temp)°C  $(clock) before duty-cycle sweep"
  env/bin/python raspi_infer.py "$GRAPH.onnx" --threads "$THREADS" \
    "${STEP_ARGS[@]+"${STEP_ARGS[@]}"}" "${RUN_ARGS[@]+"${RUN_ARGS[@]}"}" --testdata "$TESTDATA" \
      --duty_cycle "$DUTY_HOPS" --duty_seconds "$DUTY_S" --latency_out "$LATENCY_STEM"
fi

kill "$SAMPLER" 2>/dev/null || true; wait "$SAMPLER" 2>/dev/null || true
echo "--> after:  $(temp)°C  $(clock)  throttled=$(thr)   (peak during test: $(cat "$PEAK_FILE")°C)"
REMOTE

if [[ -n "$CHECK_ONLY" ]]; then
  echo "==> state only; nothing was measured"
  exit 0
fi

LAT_DIR="$DEPLOY/artifacts/pi_latency"
mkdir -p "$LAT_DIR"
scp -q "${SSH_OPTS[@]}" "$SCP_PI:~/$REMOTE_DIR/${LATENCY_STEM}*.json" "$SCP_PI:~/$REMOTE_DIR/${LATENCY_STEM}*.csv" "$LAT_DIR/"
echo "==> per-hop latency traces:"; ls -1 "$LAT_DIR/${LATENCY_STEM}"*
echo "==> done"
