from typing import TypeVar, Generic, List

import numpy as np
import copy
State = TypeVar('State') 
Action = TypeVar('Action')
class Step(object):
    def __init__(self, state_index: int, action_index: int, reward:float = 0):
        self.state = state_index
        self.action = action_index
        self.reward = reward
        pass

    def __str__(self):
        return "(" + str(self.state) + ", " + str(self.action) + ", " + str(self.reward) + ")\n"
        pass

class Trajectory(Generic[State, Action]):
    def __init__(self):
        self.steps: List[Step] = []
        # self.trajectory.append(Step(state_index, action_index))

    def append(self, state_index: int, action_index: int, reward: float = 0):
        self.steps.append(Step(state_index, action_index, reward))
    
    def append_step(self, step: Step):
        self.steps.append(Step(step.state, step.action, step.reward))

    def __str__(self):
        out_print = ""
        for step in self.steps:
            out_print += str(step)
        return out_print
    
    # We retro-propagate 
    def enrich(self, gamma: float = 0.9, alpha = 0.5):
        enriched_trajectory = Trajectory()
        # Initialisation
        current_step = copy.deepcopy(self.steps[-1])
        enriched_trajectory.append_step(current_step)
        previous_reward = current_step.reward
        for step in self.steps[-2::-1]:
            current_step = copy.deepcopy(step)
            current_step.reward = (1 - alpha) * current_step.reward + alpha * gamma * previous_reward
            enriched_trajectory.append_step(current_step)
            previous_reward = current_step.reward
        enriched_trajectory.steps = enriched_trajectory.steps[::-1]
        return enriched_trajectory
