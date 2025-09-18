#!/bin/bash
#SBATCH --job-name=idle-nodes
#SBATCH --partition=train
#SBATCH --nodes=160
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --exclusive

head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" \
         hostname -I | awk '{print $1}')
export MASTER_PORT=12357
export WORLD_SIZE=$SLURM_JOB_NUM_NODES

echo "Keeping $WORLD_SIZE nodes busy..."

srun torchrun \
   --nnodes=$SLURM_JOB_NUM_NODES \
   --nproc_per_node=1 \
   --rdzv_backend=c10d \
   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
   --rdzv_id=$SLURM_JOB_ID \
  wait.py \
   --duration 120
