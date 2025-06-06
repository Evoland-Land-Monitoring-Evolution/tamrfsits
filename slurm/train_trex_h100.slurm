#!/bin/bash
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH --partition=gpu_h100
#SBATCH --signal=SIGUSR1@90
#SBATCH --qos=gpu_h100
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00             # Walltime 24h
#SBATCH --mem-per-cpu=11G           # memory per cpu
#SBATCH --export=none              #  to start the job with a clean environnement and source of ~/.bashrc
#SBATCH --open-mode=append
cd /work/scratch/data/$USER/src/tamrfsits/
module load conda
conda activate .pixi/envs/default/
cd bin
export PYTHONOPTIMIZE=TRUE
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/$SLURM_JOBID/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True srun --export=ALL python ./train.py "$@" location=trex datamodule.config.num_workers=16 datamodule.config.cache_dir=$TMPDIR/$SLURM_JOBID/
