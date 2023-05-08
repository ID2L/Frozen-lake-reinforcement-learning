
import numpy as np
from nptyping import Float, NDArray, Shape
import random
from src.Classes.Policy import Policy
from src.Classes.Agent import Agent
from src.Functions.epsilon_greedy_policy_factory import make_epsilon_greedy_policy
from src.Functions.reshape import reshape_one
# Generic Initialisation
def MC(environment, epsilon, warmup_epoch = 500, maximum_epoch = 15000):
    # Get the observation space & the action space
    environment_space_length: int = environment.observation_space.n # type: ignore
    action_space_length: int = environment.action_space.n # type: ignore
    Q_sa = np.zeros((environment_space_length, action_space_length))
    incremental_counter = np.zeros((environment_space_length, action_space_length))
    epsilon_greedy_policy = Policy(
        make_epsilon_greedy_policy(epsilon = epsilon, Q_sa = Q_sa),
        environment_space_length,
        action_space_length
    )

    def update_MC(agent: Agent):
        # All the (state, action) pair got updated with the last reward of the run
        final_value = agent.trajectory.steps[-1].reward
        already_visited = np.full((environment_space_length, action_space_length), False)
        for step in agent.trajectory.steps:
            print("already_visited[step.state, step.action]")
            print(already_visited[step.state, step.action])
            if already_visited[step.state, step.action]: continue
            increment = incremental_counter[step.state, step.action]
            old_value = Q_sa[step.state, step.action]
            incremental_counter[step.state, step.action] += 1
            Q_sa[step.state, step.action]= (increment * old_value + final_value) / incremental_counter[step.state, step.action]
            already_visited[step.state, step.action] = True
            
    learning_agent = Agent(epsilon_greedy_policy, update_MC)
    for epoch in range(warmup_epoch):
        print("EPOCH " + str(epoch))
        core_loop(environment, learning_agent)
        learning_agent.update()

    converged = False
    difference_list = []
    previous_shape = reshape_one(Q_sa)
    loop = 0
    while not converged and loop < maximum_epoch:
        print("loop" + str(loop))
        core_loop(environment, learning_agent)
        learning_agent.update()
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

