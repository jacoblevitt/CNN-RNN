#!/bin/bash

#SBATCH --job-name=HDN
#SBATCH --output=HDN_test2.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=60000
#SBATCH --mail-user=jacob.levitt@yale.edu

module restore HDN
source activate HDN
python multi_GPU.py
tensorboard --logdir /logs --host 0.0.0.0
