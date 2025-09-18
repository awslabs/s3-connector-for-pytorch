#!/bin/bash
#SBATCH --job-name=fsdp-load
#SBATCH --partition=train
#SBATCH --nodes=200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800

head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
         hostname -I | awk '{print $1}')
export MASTER_PORT=12356
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * 1))
echo "MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$WORLD_SIZE"

srun torchrun \
   --nnodes=$SLURM_JOB_NUM_NODES \
   --nproc_per_node=2 \
   --rdzv_backend=c10d \
   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
   --rdzv_id=$SLURM_JOB_ID \
  load.py \
   --backend gloo \
   --uri ??? \
   --region ???\
   --suffix ???
