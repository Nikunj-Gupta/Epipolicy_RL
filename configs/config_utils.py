import pandas as pd
import yaml 

exps = pd.read_csv('configs/exps.csv')

config = {} 
for e in exps.to_dict(orient='records'): 
    exp_name = 'exp_' + str(e['num']) 
    e['seed'] = 10 
    config[exp_name] = e 

with open('configs/config.yaml', 'w') as file:
    yaml.dump(config, file) 

