#!/bin/bash

#SBATCH -t 3-00:00
#SBATCH -G 0
#SBATCH --mem=81920MB
#SBATCH -w discover-01

conda activate /home/tb5zhh/.conda/envs/scribblekitti/
python /home/yujc/scribble/scribblekitti/LESS.py --solve_seq=$1
