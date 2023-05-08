from typing import Callable
import gymnasium as gym
from nptyping import Float, NDArray, Shape
import numpy as np

from src.Classes.Policy import Policy
from src.Classes.Agent import Agent
from src.Functions.epsilon_greedy_policy_factory import make_epsilon_greedy_policy
from src.Functions.reshape import reshape_one
def SARSA(environment, epsilon = 0.1, alpha = 0.1, gamma = 0.99, warmup_epoch = 500, maximum_epoch = 15000):
    # Get the observation space & the action space
    environment_space_length: int = environment.observation_space.n # type: ignore
    action_space_length: int = environment.action_space.n # type: ignore
    Q_sa = np.zeros((environment_space_length, action_space_length))

    epsilon_greedy_policy = Policy(
        make_epsilon_greedy_policy(epsilon = epsilon, Q_sa = Q_sa),
        environment_space_length,
        action_space_length
    )
    def update_SARSA(agent: Agent, state_index: int, action_index: int, next_state: int, next_action: int,  reward:float = 0.0):
        # Q[s, a] := Q[s, a] + α[r + γQ(s', a') - Q(s, a)]
        Q_sa[state_index, action_index] = Q_sa[state_index, action_index] + alpha * (reward + gamma * Q_sa[next_state, next_action] - Q_sa[state_index, action_index])
        # if Q_sa[state_index, action_index] != Q_sa[state_index, action_index] + alpha * (reward + gamma * Q_sa[next_state, next_action] - Q_sa[state_index, action_index]):
        #     print(Q_sa[state_index, action_index] + alpha * (reward + gamma * Q_sa[next_state, next_action] - Q_sa[state_index, action_index]))


    learning_agent = Agent(epsilon_greedy_policy, update_SARSA)

    for epoch in range(warmup_epoch):
        core_loop(environment, learning_agent)
    
    converged = False
    difference_list = []
    previous_shape = reshape_one(Q_sa)
    loop = 0
    while not converged and loop < maximum_epoch:
        # print("loop" + str(loop))
        core_loop(environment, learning_agent)
        converged = has_converged(Q_sa, previous_shape, difference_list)
        previous_shape = reshape_one(Q_sa)
        loop += 1
    return {
        "Q_sa": Q_sa,
        "total_epoch": warmup_epoch + loop,
        "convergence_attained": converged
    }

def core_loop(environment, learning_agent):
    learning_agent.current_state_index = environment.reset()[0] # Initial state
    stop = False
    action = learning_agent.pick_next()
    while not stop:
        '''
        While Agent is not stopped
        Agent perform the action
        Agent pick an action from its state and its policy
        Agent update its trajectory: action taken, new state, new reward
        Update the stop condition if Goal reached or terminated
        '''
        observation, reward, terminated, truncated, info = environment.step(action)
        next_action_index = learning_agent.pick_next(observation)
        learning_agent.trajectory.append(observation, next_action_index, float(reward))
        learning_agent.update(
            learning_agent.current_state_index,
            action,
            observation,
            next_action_index,
            reward)
        learning_agent.current_state_index = observation
        action = next_action_index
        stop = terminated or truncated

def has_converged(Q_sa, previous_shape: NDArray,  difference_sequence: list):
    reshaped_Q_sa = reshape_one(Q_sa)
    difference_number = 0
    for row_index in range(previous_shape.shape[0]):
        is_same = (reshaped_Q_sa[row_index, ] == previous_shape[row_index, ]).all()
        if not is_same: difference_number += 1
        # print(reshaped_Q_sa[row_index, ])
        # print(previous_shape[row_index, ])
        # print(is_same)
    difference_sequence.append(difference_number)
    # We said we converge if the matrix shape didn't change over the last 100 iterations
    return len(difference_sequence) > 100 and sum(difference_sequence[-100:]) == 0






