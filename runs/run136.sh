#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=14:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=ppo_rs_262
#SBATCH --output=ppo_rs_262.out

source ../venvs/epipolicy/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 runner.py  --exp ppo_rs_262 --config configs/ppo.yaml --scenario jsons/COVID_C.json --algo ppo