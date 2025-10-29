#!/bin/bash
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH --signal=SIGUSR1@90
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00             # Walltime 24h
#SBATCH --open-mode=append
cd $WORK/src/tamrfsits/
module purge
module load python/3.11.5
conda activate .pixi/envs/default
export PYTHONOPTIMIZE=TRUE
export TORCHINDUCTOR_CACHE_DIR=$JOBSCRATCH/torchinductor_cache
export TORCH_HOME=$WORK/cache/torch
export MPLCONFIGDIR=$WORK/cache/mpl/

export UTILISE_CONFIGURATION_FILE=$WORK/src/tamrfsits/data/models/utilise/demo.yaml
export UTILISE_WEIGHTS_FILE=$WORK/src/tamrfsits/data/models/utilise/utilise_earthnet2021.pth

cd bin
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True srun --export=ALL python ./test.py "$@"
