#!/bin/bash
#SBATCH -t 5-00:00
#SBATCH -G 2
#SBATCH -w discover-01

/data14/tb5zhh/.conda/envs/scribblekitti/bin/python3 /data14/yujc/scribble/scribblekitti/train_mt_LESS.py
# python /home/yujc/scribble/scribblekitti/train_mt.py
