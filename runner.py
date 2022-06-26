# General 
import json, argparse, yaml 

# Epipolicy 
from epipolicy_environment import EpiEnv 

# Reinforcement Learning 
import torch 
from stable_baselines3.common.env_checker import check_env 
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback 


TOTAL_TIMESTEPS = 100_000

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

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
        learning_rate=config['learning_rate'], 
        buffer_size=int(config['buffer_size']), 
        learning_starts=int(config['learning_starts']), 
        batch_size=int(config['batch_size']), 
        tau=config['tau'], 
        gamma=config['gamma'], 
        train_freq=int(config['train_freq']), 
        gradient_steps=int(config['gradient_steps']), 
        action_noise=None, 
        replay_buffer_class=None, 
        replay_buffer_kwargs=None, 
        optimize_memory_usage=False, 
        ent_coef='auto', 
        target_update_interval=int(config['target_update_interval']), 
        target_entropy=config['target_entropy'], 
        use_sde=False, 
        sde_sample_freq=-1, 
        use_sde_at_warmup=False, 
        tensorboard_log='/'.join(['summaries', args.scenario.split('/')[-1].split('.')[0], args.exp]), 
        create_eval_env=False, 
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
        n_steps=2048, 
        batch_size=64, 
        n_epochs=10, 
        gamma=0.99, 
        gae_lambda=0.95, 
        clip_range=0.2, 
        clip_range_vf=None, 
        normalize_advantage=True, 
        ent_coef=0.0, 
        vf_coef=0.5, 
        max_grad_norm=0.5, 
        use_sde=False, 
        sde_sample_freq=-1, 
        target_kl=None, 
        tensorboard_log='/'.join(['summaries', args.scenario.split('/')[-1].split('.')[0], args.exp]), 
        create_eval_env=False, 
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU, 
            net_arch=[int(config['first_hidden_size']), int(config['second_hidden_size'])] 
        ), 
        verbose=0, 
        seed=None, 
        device='auto', 
        _init_setup_model=True
    )

model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    log_interval=4,
    callback=CheckpointCallback(
        save_freq=TOTAL_TIMESTEPS/10, 
        save_path='/'.join([
            'summaries', 
            args.scenario.split('/')[-1].split('.')[0], 
            args.exp, 
            'model_checkpoints_'+str(args.algo)
        ]) 
    )
)
