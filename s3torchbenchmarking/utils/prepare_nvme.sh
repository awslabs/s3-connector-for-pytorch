#!/usr/bin/env bash
#
# Mount an NVMe drive (by default, at `./nvme/`) where this script is run.

nvme_dir=${1:-"./nvme/"}

sudo umount "$nvme_dir"
sudo rm -rf "$nvme_dir"
sudo mkfs -t xfs -f /dev/nvme1n1
sudo mkdir "$nvme_dir"
sudo mount /dev/nvme1n1 "$nvme_dir"
sudo chmod 777 "$nvme_dir"
