# General 
import json, argparse, yaml 

# Epipolicy 
from epipolicy_environment import EpiEnv 

# Reinforcement Learning 
import torch 
from stable_baselines3.common.env_checker import check_env 
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback 


parser = argparse.ArgumentParser(description='PyTorch Epipolicy SAC/PPO Training')

parser.add_argument(
    '--exp', 
    # default='exp_name', 
    type=str, 
    help='path to config file for RL algorithm'
) 

parser.add_argument(
    '--config', 
    # default='configs/sample.yaml', 
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

# env = gym.make("Pendulum-v1")

env = EpiEnv(scenario) 
check_env(env)


config=None 
with open(args.config, "r") as stream:
    try: config = yaml.safe_load(stream) 
    except yaml.YAMLError as exc: print(exc) 

config = config[args.exp]


if args.algo == 'sac': 
    model = SAC(
        policy="MlpPolicy", 
        env=env, 
        learning_rate=float(config['learning_rate']), 
        buffer_size=int(float(config['buffer_size'])), 
        learning_starts=int(config['learning_starts']), 
        batch_size=int(config['batch_size']), 
        tau=float(config['tau']), 
        gamma=float(config['gamma']), 
        train_freq=int(config['train_freq']), 
        gradient_steps=int(config['gradient_steps']), 
        action_noise=None, 
        replay_buffer_class=None, 
        replay_buffer_kwargs=None, 
        optimize_memory_usage=False, 
        ent_coef='auto', 
        target_update_interval=int(config['target_update_interval']), 
        target_entropy=float(config['target_entropy']), 
        use_sde=False, 
        sde_sample_freq=-1, 
        use_sde_at_warmup=False, 
        tensorboard_log='/'.join(['summaries', args.scenario.split('/')[-1].split('.')[0], args.exp]), 
        create_eval_env=env, 
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU, 
            net_arch=[int(config['first_hidden_size']), int(config['second_hidden_size'])] 
        ), 
        verbose=1, 
        seed=config['seed'], 
        device='auto', 
        _init_setup_model=True
    ) 
elif args.algo == "ppo": 
    model = PPO(
        policy="MlpPolicy", 
        env=env, 
        learning_rate=config['learning_rate'], 
        n_steps=config['n_steps'], 
        batch_size=config['batch_size'], 
        n_epochs=config['n_steps'], 
        gamma=config['gamma'], 
        gae_lambda=config['gae_lambda'], 
        clip_range=config['clip_range'], 
        clip_range_vf=None, 
        normalize_advantage=True, 
        ent_coef=config['ent_coef'], 
        vf_coef=config['vf_coef'], 
        max_grad_norm=config['max_grad_norm'], 
        use_sde=False, 
        sde_sample_freq=-1, 
        target_kl=None, 
        tensorboard_log='/'.join(['summaries', args.scenario.split('/')[-1].split('.')[0], args.exp]), 
        create_eval_env=env, 
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU, 
            net_arch=[int(config['first_hidden_size']), int(config['second_hidden_size'])] 
        ), 
        verbose=1, 
        seed=config['seed'], 
        device='auto', 
        _init_setup_model=True
    )

# model.learn(
#     total_timesteps=config['total_timesteps'], 
#     log_interval=4,
#     callback=CheckpointCallback(
#         save_freq=config['total_timesteps']/10, 
#         save_path='/'.join([
#             'summaries', 
#             args.scenario.split('/')[-1].split('.')[0], 
#             args.exp, 
#             'model_checkpoints_'+str(args.algo)
#         ]) 
#     )
# )

model.learn(

    total_timesteps=config['total_timesteps'], 
    callback=CheckpointCallback(
        save_freq=config['total_timesteps']/10, 
        save_path='/'.join([
            'summaries', 
            args.scenario.split('/')[-1].split('.')[0], 
            args.exp, 
            'model_checkpoints_'+str(args.algo)
        ]) 
    ), 
    eval_env=env,
    eval_freq=25*52, # train for 25 years before evaluation 
    n_eval_episodes=1*52, # evaluate for 1 year 
)