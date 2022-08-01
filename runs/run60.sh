#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=4:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=ppo_new_rs_92
#SBATCH --output=ppo_new_rs_92.out

source ../venvs/epipolicy/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 runner.py  --exp ppo_new_rs_92 --config configs/ppo.yaml --scenario jsons/SIRV_B.json --algo ppo