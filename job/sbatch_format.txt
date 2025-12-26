#!/bin/bash

#SBATCH -A master
#SBATCH -p normal

# output file (capture both stdout and stderr)
#SBATCH --output=output_hpc6.log
#SBATCH --error=output_adj_hpc6.log

# Use 4 nodes
##SBATCH --nodes=4

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=hpc6
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4  # Adjust CPU allocation per task
#SBATCH -t 10-00:00:00
#SBATCH --gpu-bind=none

# Load environment
source /cluster/datastore/vemundvb/enviroments/diff_env/bin/activate

# Run python script
python -u main.py
