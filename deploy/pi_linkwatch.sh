#!/usr/bin/env bash
# Record what the Pi's own network interface is doing, on the Pi, to a file.
#
#   deploy/run_on_pi.sh copies this across; start it there with
#       nohup ~/eeg/pi_linkwatch.sh eth0 > /dev/null 2>&1 &
#   and read ~/eeg/linkwatch.log afterwards.
#
# Why a file and not ssh: the thing being diagnosed is the ssh connection
# dropping, so anything streamed over it is lost exactly when it matters. This
# writes locally once a second and survives the disconnect; the interesting
# part is the lines either side of the gap.
#
# The counter that settles it is carrier_changes from `ip -s link`.  It counts
# link up/down transitions the *Pi* saw:
#
#   it increments  -> the Pi's own PHY lost the link: cable, EEE/power saving,
#                     or the peer stopped driving it.
#   it does not    -> the Pi never saw a problem.  The drop is on the other
#                     side (a USB dongle resetting on the host, say), or the
#                     link stayed up while traffic stopped flowing -- stale
#                     IPv6 neighbour state with no IPv4 to fall back on.
#
# Also recorded: the throttle word (under-voltage is latched there and is the
# usual suspect when a board misbehaves under sustained load), SoC temperature,
# interface error/drop counters, and anything the kernel logs about the link.
set -uo pipefail

IFACE="${1:-eth0}"
LOG="${2:-$HOME/eeg/linkwatch.log}"
INTERVAL="${INTERVAL:-1}"

mkdir -p "$(dirname "$LOG")"

field() {  # field <key> -- pull one number out of `ip -s link show`
  ip -s link show "$IFACE" 2>/dev/null | tr ' ' '\n' | grep -A 1 "^$1\$" | tail -1
}

carrier_changes() {
  cat "/sys/class/net/$IFACE/carrier_changes" 2>/dev/null || echo "?"
}
operstate() { cat "/sys/class/net/$IFACE/operstate" 2>/dev/null || echo "?"; }
speed()     { cat "/sys/class/net/$IFACE/speed"     2>/dev/null || echo "?"; }

{
  echo "# linkwatch on $(hostname) iface=$IFACE started $(date -Is)"
  echo "# boot: $(uptime -s)   kernel: $(uname -r)"
  ethtool "$IFACE" 2>/dev/null | sed 's/^/# ethtool: /' || true
  ethtool --show-eee "$IFACE" 2>/dev/null | sed 's/^/# eee: /' || true
  echo "# time carrier_changes operstate speed_mbit rx_err rx_drop tx_err tx_drop throttled temp_c load1"
} >> "$LOG"

LAST_DMESG="$(dmesg 2>/dev/null | wc -l || echo 0)"

while :; do
  RX="$(ip -s link show "$IFACE" 2>/dev/null | awk '/RX:/{getline; print $3, $4}')"
  TX="$(ip -s link show "$IFACE" 2>/dev/null | awk '/TX:/{getline; print $3, $4}')"
  THR="$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2 || echo '?')"
  TEMP="$(vcgencmd measure_temp 2>/dev/null | tr -dc '0-9.' || echo '?')"
  LOAD="$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo '?')"
  echo "$(date -Is) $(carrier_changes) $(operstate) $(speed) ${RX:-? ?} ${TX:-? ?} $THR $TEMP $LOAD" >> "$LOG"

  # Anything the kernel said since the last tick -- link flaps and USB resets
  # land here, and they are what names the cause.
  NOW="$(dmesg 2>/dev/null | wc -l || echo 0)"
  if [[ "$NOW" -gt "$LAST_DMESG" ]]; then
    dmesg 2>/dev/null | tail -n "$((NOW - LAST_DMESG))" | sed 's/^/# dmesg: /' >> "$LOG"
    LAST_DMESG="$NOW"
  fi
  sleep "$INTERVAL"
done
