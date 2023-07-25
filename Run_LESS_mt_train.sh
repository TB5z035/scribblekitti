#!/bin/bash

#SBATCH -t 3-00:00
#SBATCH -G 4
#SBATCH -w discover-01

conda activate /home/tb5zhh/.conda/envs/scribblekitti/
python /home/yujc/scribble/scribblekitti/train_mt_LESS.py
# python /home/yujc/scribble/scribblekitti/train_mt.py
