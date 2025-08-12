#!/usr/bin/env bash
#
# Run dataset benchmarks.#
# Run dataset benchmarks.

# Check if multiple GPUs are available
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Multiple GPUs detected ($GPU_COUNT). Using torchrun for distributed training."
    torchrun --nproc_per_node="$GPU_COUNT" --nnodes=1 --node_rank=0 \
        ./src/s3torchbenchmarking/dataset/benchmark.py -cd conf -cn dataset +path=./nvme/ "$@"
else
    echo "Single or no GPU detected. Using regular execution."
    ./utils/run_benchmarks.sh -s dataset -d ./nvme/ "$@"
fi
