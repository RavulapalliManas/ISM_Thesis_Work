#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 22:00:57 2021

@author: dl2820
"""

import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gym
import gym_minigrid
import matplotlib.pyplot as plt

from gym_minigrid.wrappers import RGBImgPartialObsWrapper

def make_env(env_key):
    env = RGBImgPartialObsWrapper(gym.make(env_key, render_mode="rgb_array"), tile_size=1)
    #env = gym.make(env_key)
    #env.seed(seed)
    return env



def plot_env(env, highlight=True):
    
    gridView = env.render()
    
    plt.figure()
    plt.imshow(gridView)
    plt.xticks([])
    plt.yticks([])
    
    
def get_viewpoint(env, agent_pos, agent_dir):

    #jfc.
    env.env.env.agent_dir = agent_dir
    env.env.env.agent_pos = agent_pos
    
    obs = env.gen_obs()
    obs = env.observation(obs)
    
    return obs