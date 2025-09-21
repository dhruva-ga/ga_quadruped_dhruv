#!/bin/sh
# swap_can.sh — swap SocketCAN interface names can0 <-> can1
# Usage: sudo sh swap_can.sh

set -eu

need_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "Please run as root (use sudo)." >&2
    exit 1
  fi
}

exists() {
  ip link show dev "$1" >/dev/null 2>&1
}

is_up() {
  # returns 0 if admin state is UP
  ip -o link show "$1" 2>/dev/null | grep 'state UP' >/dev/null 2>&1
}

bring_down_if_up() {
  dev="$1"
  if is_up "$dev"; then
    ip link set dev "$dev" down || true
  fi
}

restore_state() {
  dev="$1"
  was_up="$2" # "1" or "0"
  if [ "$was_up" = "1" ]; then
    ip link set dev "$dev" up || true
  fi
}

main() {
  need_root

  if ! exists can0 || ! exists can1; then
    echo "Both can0 and can1 must exist. Aborting." >&2
    exit 1
  fi

  CAN0_WAS_UP=0
  CAN1_WAS_UP=0
  is_up can0 && CAN0_WAS_UP=1 || true
  is_up can1 && CAN1_WAS_UP=1 || true

  bring_down_if_up can0 || true
  bring_down_if_up can1 || true

  TMP="can_tmp_$$"
  while exists "$TMP"; do
    TMP="can_tmp_$$_$RANDOM"
  done

  # Perform the rename swap
  ip link set dev can0 name "$TMP"
  ip link set dev can1 name can0
  ip link set dev "$TMP" name can1

  # Restore previous admin states (mapped to new names)
  restore_state can1 "$CAN0_WAS_UP" # original can0 is now can1
  restore_state can0 "$CAN1_WAS_UP" # original can1 is now can0

  echo "Swap complete. Current CAN interfaces:"
  # Show brief status of just can0/can1 if present
  ip -br link show can0 2>/dev/null || true
  ip -br link show can1 2>/dev/null || true
}

main "$@"
