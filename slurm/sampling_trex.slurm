#!/bin/bash
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00             # Walltime 10h
#SBATCH --mem-per-cpu=7G           # memory per cpu
#SBATCH --export=none              #  to start the job with a clean environnement and source of ~/.bashrc
#SBATCH --open-mode=append
cd /work/scratch/data/michelj/src/tamrfsits/
module load conda
conda activate .pixi/envs/default/
cd bin
export PYTHONOPTIMIZE=TRUE
./sample_ls2s2_fusion.py "$@"
