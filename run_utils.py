import os, yaml 
from pathlib import Path 
from itertools import count 

dumpdir = "runs/" 
if not os.path.isdir(dumpdir):
    os.mkdir(dumpdir)
fixed_text = "#!/bin/bash\n"\
             "#SBATCH --nodes=1\n"\
             "#SBATCH --cpus-per-task=16 \n"\
             "#SBATCH --time=2:00:00\n"\
             "#SBATCH --mem=40GB\n"\
             "#SBATCH --gres=gpu:1\n"


config_file = "configs/config.yaml"

with open(config_file, "r") as stream:
        try: config = yaml.safe_load(stream) 
        except yaml.YAMLError as exc: print(exc) 



for scenario in ['SIRV_A']: 
    for exp in config: 
        command = fixed_text + "#SBATCH --job-name="+exp+"\n#SBATCH --output="+exp+".out\n"
        command += "\nmodule load python/intel/3.8.6\n"\
                    "module load openmpi/intel/4.0.5\n"\
                    "\nsource ./venv/bin/activate\n"\
                    "time python3 runner.py " 
        command = ' '.join([
            command, 
            '--exp', exp, 
            "--config configs/config.yaml", 
            '--scenario', 'jsons/'+scenario+'.json'
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
