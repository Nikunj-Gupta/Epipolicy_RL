from email.policy import default
import pandas as pd
import yaml, random, argparse 


parser = argparse.ArgumentParser(description='PyTorch Epipolicy SAC/PPO Config generation') 
parser.add_argument(
    '--algo', 
    # default='sac', 
    type=str, 
    required=True,
    help=' reinforcement learning algorithm'
) 
args = parser.parse_args()

default_sac = {
    'total_timesteps':1000,
    'batch_size': '256',
    'buffer_size': 100000000,
    'first_hidden_size': 256,
    'gamma': '0.9',
    'gradient_steps': '100',
    'learning_rate': 3e-3,
    'learning_starts': '10000',
    'second_hidden_size': 256,
    'seed': 10,
    'target_entropy': -200.0,
    'target_update_interval': '1',
    'tau': '0.1',
    'train_freq': 10
} 
default_ppo = {
    'total_timesteps':1000,
    'batch_size': 8 ,
    'n_steps': 128 ,
    'gamma': 0.999 ,
    'learning_rate': 2.4224480786987734e-05 ,
    'ent_coef': 5.335150728756469e-08 ,
    'clip_range': 0.2 ,
    'n_epochs': 1 ,
    'gae_lambda': 0.99 ,
    'max_grad_norm': 2 ,
    'vf_coef': 0.9741699902198256 ,
    'first_hidden_size': 256 ,
    'second_hidden_size': 256 ,
    'seed': 10
} 

default = dict(default_sac) if args.algo == 'sac' else dict(default_ppo)

config = {} 

for seed in [random.randint(1, 100) for _ in range(10)]: 
    exp_name = args.algo + "_rs_" + str(seed) 
    config[exp_name] = dict(default) 
    config[exp_name]['seed'] = seed 

with open('configs/' + args.algo + '.yaml', 'w') as file:
    yaml.dump(config, file) 
