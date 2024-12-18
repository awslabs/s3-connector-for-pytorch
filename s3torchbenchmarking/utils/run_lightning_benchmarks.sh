#!/usr/bin/env bash
#
# Run PyTorch Lightning Checkpointing benchmarks.

./utils/run_benchmarks.sh -s lightning_checkpointing -d ./nvme/ "$@"
