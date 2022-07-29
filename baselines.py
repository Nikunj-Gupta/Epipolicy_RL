# General 
import json, argparse, yaml, numpy as np, glob, os 

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


def run(scenario, baseline, path, seed=None): 
    scenario = json.loads(open('jsons/'+scenario+'.json', "r").read()) 
    env = EpiEnv(scenario) 
    if seed: 
        np.random.seed(seed)

    obs = env.reset()
    rew_arr = [] 
    action_arr = [] 
    for i in range(100_000):
        # Random Strategy 
        if baseline == 'random':
            # action = env.action_space.sample() 
            action = np.random.uniform(0, 1, env.action_space.shape[0])
        # Lax Strategy 
        elif baseline == 'lax': 
            lax_rate = vaccine_rate(TOTAL_POP, 0.9, 12, MAX_DOSES) 
            action = np.zeros(env.action_space.shape[0])
            action[0] = lax_rate
            action[1] = 1 
        # Aggressive Strategy 
        elif baseline == 'agg': 
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
        # print(action)

    action_arr = np.array(action_arr) 
    rew_arr = np.array(rew_arr) 
    print(rew_arr.shape, action_arr.shape)
    with open(path+'_rew_arr.npy', 'wb') as f: 
            np.save(f, rew_arr) 
    with open(path+'_action_arr.npy', 'wb') as f: 
            np.save(f, action_arr) 
    return rew_arr, action_arr 

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

    # run(args.scenario, args.baseline) 
    
    for scenario in ['SIRV_A', 'SIRV_B', 'SIR_A', 'SIR_B', 'COVID_A', 'COVID_B', 'COVID_C']: 
        for algo in ['random', 'lax', 'agg']: 
            print(scenario, algo)
            logs = glob.glob(os.path.join('summaries', scenario), recursive=True) 
            if algo=='random': 
                for seed in [10,12,316,109]: 
                    run(scenario, algo, path=os.path.join(logs[0], algo+'_'+str(seed)), seed=seed) 
            else: 
                run(scenario, algo, path=os.path.join(logs[0], algo)) 
            # break 
        # break 
