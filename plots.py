import os, glob, yaml, pprint, numpy as np, pandas as pd, json
from webbrowser import get 
from collections import defaultdict 
import matplotlib.pyplot as plt 
from tensorboard.backend.event_processing import event_accumulator 


# Epipolicy 
from epipolicy_environment import EpiEnv 
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy

# SIZE_GUIDANCE = {
#     'compressedHistograms': 500, 
#     'images': 4, 
#     'audio': 4, 
#     'scalars': 10000, 
#     'histograms': 1, 
# }

STORE_EVERYTHING_SIZE_GUIDANCE = {
    'compressedHistograms': 0, 
    'images': 0, 
    'audio': 0, 
    'scalars': 0, 
    'histograms': 0, 
} 


SMOOTH = 3 
CUT = 175
MEAN_CUT = -10 

def get_values(filename, scalar="Episodic_Reward"): 
    ea = event_accumulator.EventAccumulator(filename, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
    ea.Reload() 
    # print(ea.Tags())
    ea_scalar = ea.Scalars(tag=scalar) 
    ea_scalar = pd.DataFrame(ea_scalar) 
    return ea_scalar 


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def save_npy(log_dir): 
    logs = glob.glob(os.path.join(log_dir, '**/event*'), recursive=True) 
    for log in logs: 
        path = '/'.join(log.split('/')[:-1]) 
        with open(path+'/arr.npy', 'wb') as f: 
            np.save(f, get_values(log, scalar="rollout/ep_rew_mean")['value'].to_numpy()) 

def merge(logs): 
    # pprint.pprint(logs) 
    vals = [] 
    for l in logs: 
        vals.append(np.load(l)[:CUT]) 
    vals = np.array(vals) 
    # print(vals.shape) 
    # print(vals) 
    val_means = np.array(vals).mean(axis=0)
    val_stds = np.array(vals).std(axis=0)
    # return val_means, val_stds, np.mean(val_means[MEAN_CUT:])/1e6, np.mean(val_stds[MEAN_CUT:])/1e6 
    return val_means, val_stds, np.mean(val_means), np.mean(val_stds) 
def merge_baselines(logs): 
    # pprint.pprint(logs) 
    vals = [] 
    for l in logs: 
        vals.append(np.load(l)) 
    vals = np.array(vals) 
    val_means = np.array(vals).mean(axis=0)
    val_stds = np.array(vals).std(axis=0)
    val_means = np.take(val_means, list(range(0, len(val_means), 500)))
    val_stds = np.take(val_stds, list(range(0, len(val_stds), 500)))
    print(val_means.shape) 
    print(val_stds.shape) 

    # return val_means, val_stds, np.mean(val_means[MEAN_CUT:])/1e6, np.mean(val_stds[MEAN_CUT:])/1e6 
    return val_means, val_stds, np.mean(val_means), np.mean(val_stds) 


def plot(log_dir, only_baselines=False): 

    for scenario in ['SIRV_A', 'SIRV_B', 'SIR_A', 'SIR_B', 'COVID_A', 'COVID_B', 'COVID_C']: 
        fig, ax = plt.subplots()
        fig.canvas.draw()
        if not only_baselines: 
            for algo in ['ppo', 'sac']: 
                logs = glob.glob(os.path.join(log_dir, scenario, "*"+algo+"*", '*/arr.npy'), recursive=True) 
                val_means, val_stds, mean_rew, mean_std = merge(logs) 
                print(scenario, algo, "mean: ", mean_rew, "mean_std: ", mean_std) 
                val_means = smooth(val_means, box_pts=SMOOTH)
                val_stds = smooth(val_stds, box_pts=SMOOTH)
                plt.plot(val_means[SMOOTH:][:-SMOOTH], label=algo)
                plt.fill_between(np.arange(1, len(val_means)+1), 
                        val_means - val_stds, 
                        val_means + val_stds, 
                        alpha=0.2) 
                # break 
        for algo in ['random', 'lax', 'agg']: 
            logs = glob.glob(os.path.join(log_dir, scenario, algo+"*rew_arr*"), recursive=True) 
            if algo=='random': 
                val_means, val_stds, mean_rew, mean_std = merge_baselines(logs) 
                print(scenario, algo, "mean: ", mean_rew, "mean_std: ", mean_std) 
                val_means = smooth(val_means, box_pts=SMOOTH)
                val_stds = smooth(val_stds, box_pts=SMOOTH)
                plt.plot(val_means[SMOOTH:][:-SMOOTH], label=algo)
                plt.fill_between(np.arange(1, len(val_means)+1), 
                        val_means - val_stds, 
                        val_means + val_stds, 
                        alpha=0.2) 
            else: 
                vals = np.load(logs[0]) 
                vals = np.take(vals, list(range(0, len(vals), 500)))
                vals = smooth(vals, box_pts=SMOOTH)
                plt.plot(vals[SMOOTH:][:-SMOOTH], label=algo)
                # print(vals.shape, "mean: ", np.mean(val_means)) 
        ax.set_xticklabels([int(i*500) for i in ax.get_xticks().tolist()[1:]]) 
        plt.xlim([0, CUT])
        plt.title(scenario)
        plt.legend() 
        plt.show() 
        # plt.savefig('plots/'+scenario+'.png') 
        # plt.close() 
        # break 
        
def evaluate(scenario, algo, model_path): 
    env = EpiEnv(json.loads(open('jsons/'+scenario+'.json', "r").read())) 
    # Load the trained agent
    model = PPO.load(model_path, env=env) if algo=='ppo' else SAC.load(model_path, env=env)
    print(model)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    print(mean_reward, std_reward) 
    # Enjoy trained agent 
    obs = env.reset()
    action_arr = [] 
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True) 
        action_arr.append(action) 
        obs, rewards, dones, info = env.step(action)
    action_arr = np.array(action_arr) 
    for i in range(action_arr.shape[1]): 
        plt.plot(action_arr[:,i], label='intervention '+str(i)) 

    # plt.plot(action_arr[:,0], label='intervention 1') 
    # plt.plot(action_arr[:,1], label='intervention 2') 
    plt.title(scenario+'_'+algo)
    plt.legend() 
    # plt.show() 
    print('plots/'+scenario+'_'+model_path.split('/')[-3]+'.png')
    plt.savefig('plots/intervention_'+scenario+'_'+model_path.split('/')[-3]+'.png') 
    plt.close() 
    return mean_reward, std_reward 



def plot_plans(log_dir): 
    for scenario in ['SIRV_A', 'SIRV_B', 'SIR_A', 'SIR_B', 'COVID_A', 'COVID_B', 'COVID_C']: 
        for algo in ['ppo', 'sac']: 
            logs = glob.glob(os.path.join(log_dir, scenario, "*"+algo+"*", 'model_checkpoints*/*80000*.zip'), recursive=True) 
            print(scenario, algo, logs)
            for model in logs: 
                evaluate(scenario, algo, model) 


if __name__ == '__main__': 
    # save_npy('./summaries') 
    
    plot('./summaries', only_baselines=True) 

    # plot_plans('./summaries') 

    # debug('./summaries') 