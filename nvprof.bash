#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=nvprof
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --output=nvprof.%j.out

cd /scratch/$USER/HPC/project/

set -o xtrace
nvprof --metrics all ./mnist-cnn-gpu 2
