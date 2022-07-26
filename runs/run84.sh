#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=14:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=sac_rs_4
#SBATCH --output=sac_rs_4.out

source ../venvs/epipolicy/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 runner.py  --exp sac_rs_4 --config configs/sac.yaml --scenario jsons/COVID_A.json --algo sac