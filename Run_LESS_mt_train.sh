#!/bin/bash

#SBATCH -t 5-00:00
#SBATCH -G 4
#SBATCH -w discover-01

python /home/yujc/scribble/scribblekitti/train_mt_LESS.py
# python /home/yujc/scribble/scribblekitti/train_mt.py
