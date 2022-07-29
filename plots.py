import os, glob, yaml, pprint, numpy as np, pandas as pd
from webbrowser import get 
from collections import defaultdict 
import matplotlib.pyplot as plt 
from tensorboard.backend.event_processing import event_accumulator


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
    return val_means, val_stds, np.mean(val_means[MEAN_CUT:])/1e6, np.mean(val_stds[MEAN_CUT:])/1e6 
    # return val_means, val_stds, np.mean(val_means), np.mean(val_stds) 



def plot(log_dir): 

    for scenario in ['SIRV_A', 'SIRV_B', 'SIR_A', 'SIR_B', 'COVID_A', 'COVID_B', 'COVID_C']: 
        fig, ax = plt.subplots()
        fig.canvas.draw()
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
        ax.set_xticklabels([int(i*500) for i in ax.get_xticks().tolist()[1:]]) 
        plt.xlim([0, CUT])
        plt.title(scenario)
        plt.legend() 
        # plt.show() 
        plt.savefig('plots/'+scenario+'.png') 
        plt.close() 

            
if __name__ == '__main__': 
    # save_npy('./summaries') 
    
    plot('./summaries') 