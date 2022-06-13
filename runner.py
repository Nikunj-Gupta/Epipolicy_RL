# General 
import numpy as np, json, argparse, yaml 

# Epipolicy 
# from epipolicy_environment import EpiEnv 

# Reinforcement Learning 
import gym 
from stable_baselines3.common.env_checker import check_env 
from stable_baselines3 import SAC 



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument(
    '--exp', 
    default='exp_name', 
    type=str, 
    help='path to config file for RL algorithm'
) 

parser.add_argument(
    '--config', 
    default='configs/sample.yaml', 
    type=str, 
    help='path to config file for RL algorithm'
) 

parser.add_argument(
    '--scenario', 
    default='jsons/warmup_scenario.json', 
    type=str, 
    help='path to json file for scenario'
) 


args = parser.parse_args()

scenario = json.loads(open(args.scenario, "r").read()) 

# env = EpiEnv(scenario) 
env = gym.make("Pendulum-v1")
check_env(env)


config=None 
with open(args.config, "r") as stream:
    try: config = yaml.safe_load(stream) 
    except yaml.YAMLError as exc: print(exc) 

config = config[args.exp]

model = SAC(
    policy="MlpPolicy", 
    env=env, 
    learning_rate=config['learning_rate'], 
    buffer_size=config['buffer_size'], 
    learning_starts=config['learning_starts'], 
    batch_size=config['batch_size'], 
    tau=config['tau'], 
    gamma=config['gamma'], 
    train_freq=config['train_freq'], 
    gradient_steps=config['gradient_steps'], 
    action_noise=None, 
    replay_buffer_class=None, 
    replay_buffer_kwargs=None, 
    optimize_memory_usage=False, 
    ent_coef='auto', 
    target_update_interval=config['target_update_interval'], 
    target_entropy=config['target_entropy'], 
    use_sde=False, 
    sde_sample_freq=-1, 
    use_sde_at_warmup=False, 
    tensorboard_log='/'.join(['summaries', args.scenario.split('/')[-1].split('.')[0], args.exp]), 
    create_eval_env=False, 
    policy_kwargs=None, 
    verbose=1, 
    seed=config['seed'], 
    device='auto', 
    _init_setup_model=True
)

model.learn(total_timesteps=1000, log_interval=4)


model.save('/'.join(['summaries', args.scenario.split('/')[-1].split('.')[0], args.exp, args.exp])) 