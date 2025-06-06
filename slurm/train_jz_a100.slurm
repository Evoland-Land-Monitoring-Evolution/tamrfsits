#!/bin/bash
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH --signal=SIGUSR1@90
#SBATCH -C a100
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00             # Walltime 24h
#SBATCH --open-mode=append
cd $WORK/src/tamrfsits/
module purge
module load python/3.11.5
conda activate .pixi/envs/default
export PYTHONOPTIMIZE=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$JOBSCRATCH/torchinductor_cache
export TORCH_HOME=$WORK/cache/torch
export PYTORCH_KERNEL_CACHE_PATH=$WORK/cache/torch/kernel
cd bin
srun --export=ALL python -u ./train.py "$@" location=jz datamodule.config.num_workers=8 datamodule.config.cache_dir="$JOBSCRATCH/cache"
