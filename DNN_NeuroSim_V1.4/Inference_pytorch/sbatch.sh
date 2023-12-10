#!/bin/bash
#SBATCH -A m4271
#SBATCH -C gpu
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH --exclusive

export SLURM_CPU_BIND="cores"
python inference.py --dataset cifar10 --model VGG8 --mode WAGE --inference 1 --cellBit 1 --subArray 64
