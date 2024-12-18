#!/usr/bin/env bash
#
# Mount an NVMe drive (by default, at `./nvme/`, relative to where this script is run). If a drive is already mounted at
# the specified location, clear its content.

nvme_dir=${1:-"./nvme/"} # default value

if ! mountpoint -q "$nvme_dir"; then
  rm -rf "$nvme_dir"
  mkdir -p "$nvme_dir"

  if grep -q 'NAME="Amazon Linux"' /etc/os-release; then
    sudo mkfs -t xfs /dev/nvme1n1
    sudo mount /dev/nvme1n1 "$nvme_dir"
  elif grep -q 'NAME="Ubuntu"' /etc/os-release; then
    sudo mount /dev/mapper/vg.01-lv_ephemeral "$nvme_dir"
  fi

  sudo chmod 777 "$nvme_dir"
fi
