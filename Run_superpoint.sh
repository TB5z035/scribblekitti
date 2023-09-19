#!/bin/bash

#SBATCH -t 3-00:00
#SBATCH -G 6
#SBATCH -w discover-03

/data14/yujc/.conda/envs/spt/bin/python3 /data14/yujc/scribble/scribblekitti/Superpoints_Augmented_LESS.py --solve_seq=$1
