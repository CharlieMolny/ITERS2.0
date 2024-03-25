import gym
import highway_env
from stable_baselines3 import DQN
from matplotlib import pyplot as plt
from src.envs.custom.highway import CustomHighwayEnv
from src.util import seed_everything, load_config

from highway_env import utils
import random
import numpy as np
import copy
from highway_env.envs import highway_env
from highway_env.vehicle.controller import ControlledVehicle
import torch
from src.feedback.feedback_processing import encode_trajectory

# Initialize the environment
env=highway_env.HighwayEnvFast()
# env = CustomHighwayEnv(shaping=False, time_window=5,run_tailgaiting=False,run_speed=False)
# env_config_path = 'config/env/{}{}.json'.format('highway','','')
# env_config = load_config(env_config_path)

# env.config['right_lane_reward'] = env_config['right_lane_reward']
# env.config['lanes_count'] = env_config['lanes_count']
# obs=env.reset()
# env.set_true_reward(env_config['true_reward_func'])

# Load the saved model
model = DQN.load(r'C:\Users\charl\Desktop\Dissertation\Technical Part\RePurpose_iters\trained_models\highway\regular_best_summary_expl\seed_0_lmbda_1_epsilon_1_iter_17.zip')

# Render function to display the environment

# Run a single episode
while True:
    done = False
    obs=env.reset()
    
    while not done:
        # Assuming your model has a predict function to get the action
        # You might need to preprocess the observation based on your model's requirement
        action, _ = model.predict(obs, deterministic=True)
        _, _, done, _ = env.step(action)
        # Render the current state
        env.render() 

    env.close()
