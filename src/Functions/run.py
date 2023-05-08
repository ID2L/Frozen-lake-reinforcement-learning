import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt

from typing import Callable
from src.Classes.Agent import Agent

def run_static(environment: gym.Env, agent: Agent, renderer: None | Callable[[gym.Env, int], None] = None):
    agent.current_state_index = environment.reset()[0] # Initial state
    stop = False    
    while not stop:
        '''
        While Agent is not stopped
        Agent perform the action
        Agent pick an action from its state and its policy
        Agent update its trajectory: action taken, new state, new reward
        Update the stop condition if Goal reached or terminated
        '''
        next_action_index = agent.pick_next()
        observation, reward, terminated, truncated, info = environment.step(next_action_index)
        if renderer is not None: 
            renderer(environment, next_action_index)
        agent.trajectory.append(observation, next_action_index, float(reward))
        agent.current_state_index = observation
        stop = terminated or truncated
    return agent