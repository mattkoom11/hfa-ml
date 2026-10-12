#!/usr/bin/env bash
# One-time setup: mount the NAS photo share read-write on first access, at boot or after sleep.
# Run: sudo bash scripts/setup_nas_automount.sh
# Stores the NAS password in a root-only credentials file (mode 600); it is never echoed.
set -euo pipefail

SHARE="//dxp4800plus-f84/garment-capture-media"   # on Tailscale away from home: //100.85.122.5/garment-capture-media
MOUNT="/mnt/nas-garment-capture"
CRED="/etc/samba/nas-garment-capture.cred"
USER_NAME="${SUDO_USER:?run with sudo from your own account}"
UID_N="$(id -u "$USER_NAME")"; GID_N="$(id -g "$USER_NAME")"

read -rp "NAS username [matthewkim]: " NAS_USER; NAS_USER="${NAS_USER:-matthewkim}"
read -rsp "NAS password: " NAS_PASS; echo

install -d -m 755 /etc/samba "$MOUNT"
umask 077
printf 'username=%s\npassword=%s\n' "$NAS_USER" "$NAS_PASS" > "$CRED"
chmod 600 "$CRED"; chown root:root "$CRED"
unset NAS_PASS

LINE="$SHARE $MOUNT cifs credentials=$CRED,uid=$UID_N,gid=$GID_N,vers=3.1.1,rw,noauto,x-systemd.automount,x-systemd.idle-timeout=15min,x-systemd.mount-timeout=20,_netdev,nofail 0 0"
cp /etc/fstab "/etc/fstab.bak-$(date +%F)"
grep -v " $MOUNT cifs " /etc/fstab > /etc/fstab.new || true
echo "$LINE" >> /etc/fstab.new
mv /etc/fstab.new /etc/fstab

umount "$MOUNT" 2>/dev/null || true
systemctl daemon-reload
systemctl restart "$(systemd-escape -p --suffix=automount "$MOUNT")"
ls "$MOUNT" >/dev/null && echo "OK: $MOUNT mounts on access (read-write). fstab backup: /etc/fstab.bak-$(date +%F)"
