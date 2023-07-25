#!/bin/bash

#SBATCH -t 3-00:00
#SBATCH -G 0
#SBATCH -o slurm_0719_dist_3.o
#SBATCH --mem=100000MB
#SBATCH -w discover-01

conda activate /home/tb5zhh/.conda/envs/scribblekitti/
python /home/yujc/scribble/scribblekitti/LESS.py
