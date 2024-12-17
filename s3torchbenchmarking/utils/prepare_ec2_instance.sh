#!/usr/bin/env bash
#
# Script to prepare an EC2 instance for PyTorch benchmarks. Like other scripts within this directory, it is assumed
# that this is run from within the "s3-connector-for-pytorch/s3torchbenchmarking" directory.

set -eou pipefail

# Sanity check + install Mountpoint for Amazon S3
if [[ -n $(which yum) ]]; then
  sudo yum -y upgrade

  wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.rpm
  sudo yum install -y ./mount-s3.rpm && rm ./mount-s3.rpm
elif [[ -n $(which apt) ]]; then
  sudo apt -y upgrade

  wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb
  sudo apt install -y ./mount-s3.deb && rm ./mount-s3.deb
fi

# Install s3torchconnector and s3torchbenchmarking
pip install 's3torchconnector[lightning,dcp]'
pip install -e .
