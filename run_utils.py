import os, yaml, argparse
from pathlib import Path 
from itertools import count 

parser = argparse.ArgumentParser(description='PyTorch Epipolicy SAC/PPO runs generation') 
parser.add_argument(
    '--algo', 
    # default='sac', 
    type=str, 
    required=True,
    help=' reinforcement learning algorithm'
) 
args = parser.parse_args()


dumpdir = "runs/" 
if not os.path.isdir(dumpdir):
    os.mkdir(dumpdir)
fixed_text = "#!/bin/bash\n"\
             "#SBATCH --nodes=1\n"\
             "#SBATCH --cpus-per-task=16 \n"\
             "#SBATCH --time=14:00:00\n"\
             "#SBATCH --mem=20GB\n"\
            #  "#SBATCH --gres=gpu:1\n"


config_file = "configs/sac.yaml" if args.algo == 'sac' else "configs/ppo.yaml"

with open(config_file, "r") as stream:
        try: config = yaml.safe_load(stream) 
        except yaml.YAMLError as exc: print(exc) 



# for scenario in ['SIRV_A', 'SIRV_B', 'SIR_A', 'SIR_B']: 
for scenario in ['COVID_A', 'COVID_B', 'COVID_C']: 
    for exp in config: 
        command = fixed_text + "#SBATCH --job-name="+exp+"\n#SBATCH --output="+exp+".out\n"
        command += "\nsource ../venvs/epipolicy/bin/activate\n"\
            "\nmodule load python/intel/3.8.6\n"\
            "module load openmpi/intel/4.0.5\n"\
            "time python3 runner.py " 
        command = ' '.join([
            command, 
            '--exp', exp, 
            "--config", config_file, 
            '--scenario', 'jsons/'+scenario+'.json', 
            '--algo', args.algo 
        ]) 
        # print(command) 
        log_dir = Path(dumpdir)
        for i in count(1):
            temp = log_dir/('run{}.sh'.format(i)) 
            if temp.exists():
                pass
            else:
                with open(temp, "w") as f:
                    f.write(command) 
                log_dir = temp
                break 
