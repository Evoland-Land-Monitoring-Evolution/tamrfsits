#!/bin/bash
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH --signal=SIGUSR1@90
#SBATCH -C h100
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --cpus-per-task=24
#SBATCH --time=100:00:00             # Walltime 24h
#SBATCH --open-mode=append
cd $WORK/src/tamrfsits/
module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate .pixi/envs/default
export PYTHONOPTIMIZE=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$JOBSCRATCH/torchinductor_cache
export TORCH_HOME=$WORK/cache/torch
export PYTORCH_KERNEL_CACHE_PATH=$WORK/cache/torch/kernel
cd bin
srun --export=ALL python -u ./train.py "$@" location=jz datamodule.config.num_workers=24 datamodule.config.cache_dir="$JOBSCRATCH/cache"
