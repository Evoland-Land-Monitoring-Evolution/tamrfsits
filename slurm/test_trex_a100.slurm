#!/bin/bash
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH --partition=gpu_a100
#SBATCH --signal=SIGUSR1@90
#SBATCH --qos=gpu_all
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00             # Walltime 24h
#SBATCH --mem-per-cpu=7G           # memory per cpu
#SBATCH --export=none              #  to start the job with a clean environnement and source of ~/.bashrc
cd /work/scratch/data/$USER/src/tamrfsits/
module load conda
conda activate .pixi/envs/default/
cd bin
export PYTHONOPTIMIZE=TRUE
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/$SLURM_JOBID/
export UTILISE_CONFIGURATION_FILE=/work/scratch/data/michelj/src/tamrfsits/data/models/utilise/demo.yaml
export UTILISE_WEIGHTS_FILE=/work/scratch/data/michelj/src/tamrfsits/data/models/utilise/utilise_earthnet2021.pth

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True srun --export=ALL python ./test.py "$@"
