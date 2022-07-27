# General 
import json, argparse, yaml, numpy as np 

# Epipolicy 
from epipolicy_environment import EpiEnv 

# Reinforcement Learning 
import torch 
from stable_baselines3.common.env_checker import check_env 
from stable_baselines3.common.evaluation import evaluate_policy

TOTAL_POP = 2224526
MAX_DOSES = 9200


def vaccine_rate(total_pop, pc_pop, nmonths, max_doses):
    return total_pop * pc_pop / (nmonths * 30) / max_doses


if __name__ == '__main__': 
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
        # Random Strategy 
        if args.baseline == 'random':
            action = env.action_space.sample() 
        # Lax Strategy 
        elif args.baseline == 'lax': 
            lax_rate = vaccine_rate(TOTAL_POP, 0.9, 12, MAX_DOSES) 
            action = np.zeros(env.action_space.shape[0])
            action[0] = lax_rate
            action[1] = 1 
        # Aggressive Strategy 
        elif args.baseline == 'agg': 
            agg_rate = vaccine_rate(TOTAL_POP, 0.85, 9, MAX_DOSES) 
            action = np.zeros(env.action_space.shape[0]) 
            if i <= 9*30:
                action[0] = agg_rate
            action[1] = 0.8
            if i <= 4*30:
                action[2:] = 1
            else:
                action[2:] = 0.5
        action_arr.append(action) 
        obs, rewards, dones, info = env.step(action)
        rew_arr.append(rewards) 
        print(action)

    action_arr = np.array(action_arr) 
    rew_arr = np.array(rew_arr) 
    print(rew_arr)