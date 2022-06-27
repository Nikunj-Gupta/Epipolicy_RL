#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=exp_16
#SBATCH --output=exp_16.out

source ../venvs/epipolicy/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 runner.py  --exp exp_16 --config configs/config_new.yaml --scenario jsons/SIRV_A.json --algo sac