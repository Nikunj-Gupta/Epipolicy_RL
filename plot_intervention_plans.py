# General 
import json, argparse, yaml, numpy as np 
import matplotlib.pyplot as plt 


# Epipolicy 
from epipolicy_environment import EpiEnv 

# Reinforcement Learning 
import torch 
from stable_baselines3.common.env_checker import check_env 
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback 
from stable_baselines3.common.evaluation import evaluate_policy



parser = argparse.ArgumentParser(description='PyTorch Epipolicy SAC/PPO Training')

parser.add_argument(
    '--model_path', 
    default="summaries/SIRV_A/ppo_rs_64/model_checkpoints_ppo/rl_model_100000_steps.zip", 
    type=str, 
    help='path to config file for RL algorithm'
) 

parser.add_argument(
    '--scenario', 
    # default='jsons/warmup_scenario.json', 
    type=str, 
    help='path to json file for scenario'
) 

parser.add_argument(
    '--algo', 
    # default='sac', 
    type=str, 
    help=' reinforcement learning algorithm'
) 

args = parser.parse_args()

scenario = json.loads(open(args.scenario, "r").read()) 


env = EpiEnv(scenario) 
# Load the trained agent
model = PPO.load(args.model_path, env=env)

print(model)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(mean_reward, std_reward) 

# Enjoy trained agent 
obs = env.reset()
action_arr = [] 
for i in range(52):
    # # Random Strategy 
    # action = env.action_space.sample()
    action, _states = model.predict(obs, deterministic=True) 
    action_arr.append(action) 
    obs, rewards, dones, info = env.step(action)
    # env.render() 
    print(action)

action_arr = np.array(action_arr) 
plt.plot(action_arr[:,0], label='intervention 1') 
plt.plot(action_arr[:,1], label='intervention 2') 
plt.legend() 
plt.show() 

