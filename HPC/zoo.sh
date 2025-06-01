#!/bin/bash -l
#SBATCH --job-name=test # Name of the job that will appear in the HPC logs
#SBATCH --output=output.log # Here put where the output will be re-directed
#SBATCH --error=error.log # Here put where the errors will be re-directed
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH --time=0-48:00:00 # Max runtime (DD-hh:mm:ss)
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --mem=64GB

micromamba activate thesis_env_310 # Name of our virtual environment
CUDA_VISIBLE_DEVICES=0 python /home/users/atodorova/alisa-thesis/malnet-cnn/scripts/test.py #Here put the path to the Python script you want to run