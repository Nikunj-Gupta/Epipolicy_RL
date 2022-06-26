import pandas as pd
import yaml, random 

exps = pd.read_csv('configs/exps.csv') 

default={
    'learning_rate':[3e-3, 3e-4, 3e-5],
    'buffer_size':[1000000, 10000000, 100000000],
    'learning_starts':[1000, 10000, 100000],
    'batch_size':[128, 256, 512],
    'tau':[0.05, 0.1, 0.2],
    'gamma':[0.99, 0.95, 0.9],
    'train_freq':[1, 10, 100, 1000, 10000],
    'gradient_steps':[1, 10, 100],
    'target_update_interval':[1, 5, 10],
    'target_entropy':[-0.2, -2, -20, -200, -2000],
    'first_hidden_size':[64, 128, 256, 512, 1024],
    'second_hidden_size':[64, 128, 256, 512, 1024], 
    'seed': [10, 12, 20, 90, 100] 
}

config = {} 
for e in exps.to_dict(orient='records'): 
    exp_name = 'exp_' + str(e['num']) 
    e['seed'] = random.choice(default['seed']) 
    config[exp_name] = e 

for exp in config: 
    if 'X' in config[exp].values(): 
        for key in config[exp]: 
            if config[exp][key] == 'X': 
                config[exp][key] = random.choice(default[key]) 



with open('configs/config_new.yaml', 'w') as file:
    yaml.dump(config, file) 

