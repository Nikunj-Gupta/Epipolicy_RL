# General 
import json, argparse, yaml, numpy as np 

# Epipolicy 
from epipolicy_environment import EpiEnv 

# Reinforcement Learning 
import torch 
from stable_baselines3.common.env_checker import check_env 
from stable_baselines3.common.evaluation import evaluate_policy

parser = argparse.ArgumentParser(description='PyTorch Epipolicy SAC/PPO Training')

parser.add_argument(
    '--scenario', 
    # default='jsons/warmup_scenario.json', 
    type=str, 
    help='path to json file for scenario'
) 

parser.add_argument(
    '--baseline', 
    # default='random', 
    type=str, 
    help=' baseline algorithm'
) 

args = parser.parse_args()

scenario = json.loads(open(args.scenario, "r").read()) 


env = EpiEnv(scenario) 

obs = env.reset()

rew_arr = [] 
action_arr = [] 
for i in range(1000):
    # # Random Strategy 
    if args.baseline == 'random':
        action = env.action_space.sample()
    action_arr.append(action) 
    obs, rewards, dones, info = env.step(action)
    rew_arr.append(rewards) 
    print(action)

action_arr = np.array(action_arr) 
rew_arr = np.array(rew_arr) 
print(rew_arr)