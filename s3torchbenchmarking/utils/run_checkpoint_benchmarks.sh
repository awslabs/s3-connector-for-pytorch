#!/usr/bin/env bash
#
# Run PyTorch Checkpointing benchmarks.

./utils/run_benchmarks.sh -s pytorch_checkpointing -d ./nvme/ "$@"
