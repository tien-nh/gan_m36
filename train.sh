#!/bin/bash
#SBATCH --job-name=gan       # Job name
#SBATCH --output=slurm/gan.txt      # Output file
#SBATCH --error=slurm/gan_error.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=80G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

python evaluate_model.py