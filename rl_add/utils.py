"""
Utility functions to load models
"""

import gym
from gym import envs

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import DQN, A2C, PPO, DDPG
from stable_baselines3.common.monitor import Monitor

POLICY = 'MlpPolicy'


# register continuous action space version of cart pole environment
envs.register(
    id='CartPoleC-v1', 
    entry_point="rl_add.envs:CartPoleCont", 
    max_episode_steps=500,
    reward_threshold=475.0,
)


def make_env(env_id: str, rank: int, seed: int = 0, gravity_factor=1):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.env.gravity = env.env.gravity * gravity_factor
        env.seed(seed + rank)
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init


def get_model(mtype, gravity_factor=1):
    if mtype == 'DQN':
        mon = SubprocVecEnv([
            make_env('CartPole-v1', i, gravity_factor=gravity_factor) 
            for i in range(1)
        ])
        return DQN(
            POLICY, mon, verbose=0, 
            learning_rate=2.3e-3, 
            batch_size=64, 
            buffer_size=100000, 
            learning_starts=1000,
            gamma=0.99, 
            target_update_interval=10, 
            train_freq=256, 
            gradient_steps=128, 
            exploration_fraction=0.16, 
            exploration_final_eps=0.04, 
            policy_kwargs={'net_arch': [256, 256]}
        ), mon
    elif mtype == 'A2C':
        mon = SubprocVecEnv([
            make_env('CartPole-v1', i, gravity_factor=gravity_factor) 
            for i in range(8)
        ])
        return A2C(
            POLICY, mon, verbose=0, 
            ent_coef=0.0, 
        ), mon
    elif mtype == 'PPO':
        mon = SubprocVecEnv([
            make_env('CartPole-v1', i, gravity_factor=gravity_factor) 
            for i in range(8)
        ])
        return PPO(
            POLICY, mon, 
            verbose=0, 
            n_steps=32, 
            batch_size=256, 
            gae_lambda=0.8, 
            gamma=0.98, 
            n_epochs=20, 
            ent_coef=0.0, 
            learning_rate=0.001, 
            clip_range=0.2,
        ), mon
    elif mtype == 'DDPG':
        mon = SubprocVecEnv([
            make_env('CartPoleC-v1', i, gravity_factor=gravity_factor) 
            for i in range(1)
        ])
        return DDPG(POLICY, mon, verbose=0), mon
    raise ValueError(f"Unknown model: {mtype}")


def get_eval_env(mtype, gravity_factor=1.0):
    # get a cartpole environment for evaluation with a particular gravity factor
    if mtype in ['DQN', 'A2C', 'PPO']:
        env = gym.make('CartPole-v1')
        env.env.gravity = env.env.gravity * gravity_factor
        return Monitor(env)
    elif mtype == 'DDPG':
        env = gym.make('CartPoleC-v1')
        env.env.gravity = env.env.gravity * gravity_factor
        return Monitor(env)
    raise ValueError(f"Unknown model: {mtype}")