#!/usr/bin/env bash
#
# Run dataset benchmarks.

# TODO: see if it can reuse the `run_benchmarks.sh` script template here
python ./src/s3torchbenchmarking/dataset/benchmark.py -cd conf -cn dataset "$@"
