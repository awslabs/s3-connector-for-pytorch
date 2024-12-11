#!/usr/bin/env bash
#
# Script to prepare an EC2 instance for PyTorch benchmarks.

set -euo pipefail

# Sanity check
sudo yum -y upgrade # OR, `sudo apt -y upgrade`

# Activate the default PyTorch env
#
# This command only works on specific AMI (DL-AMI): make sure to check the compatibility between the latter and the EC2
# instance type currently in use (if incompatible, the command will either fail or generate a warning).
source activate pytorch

# Install Rust (https://www.rust-lang.org/learn/get-started)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"

# Addresses error "RuntimeError: operator torchvision::nms does not exist" while trying the run the benchmarks
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Addresses error "TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'" while
# trying to install s3torchbenchmarking
pip install "setuptools<71"

# Required to build s3torchconnectorclient (see its README)
sudo yum -y install cmake3 clang

pip install 's3torchconnector[lightning,dcp]'
cd s3torchbenchmarking && pip install -e .
