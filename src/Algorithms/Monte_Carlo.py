from src.Classes.Policy import Policy
from src.Classes.Agent import Agent

import numpy as np
import random

# Generic Initialisation
def MC(environment, epoch_number = 8000):
    # Get the observation space & the action space
    environment_space_length: int = environment.observation_space.n # type: ignore
    action_space_length: int = environment.action_space.n # type: ignore
    Q_sa = np.zeros((environment_space_length, action_space_length))
    incremental_counter = np.zeros((environment_space_length, action_space_length))
    random_policy = Policy(
        lambda agent, state: 
            random.randint(0, action_space_length - 1)
        ,
        environment_space_length,
        action_space_length
    )
    def update_MC(agent: Agent):
        # All the (state, action) pair got updated with the last reward of the run
        final_value = agent.trajectory.steps[-1].reward
        for step in agent.trajectory.steps:
            increment = incremental_counter[step.state, step.action]
            old_value = Q_sa[step.state, step.action]
            incremental_counter[step.state, step.action] += 1
            Q_sa[step.state, step.action]= (increment * old_value + final_value) / incremental_counter[step.state, step.action]
            
    
    learning_agent = Agent(random_policy, update_MC)
    
    for epoch in range(epoch_number):
        stop = False
        environment.reset()
        learning_agent.current_state_index = environment.reset()[0] # Initial state
        action = learning_agent.pick_next()
        while not stop:
            '''
            While Agent is not stopped:
                Agent perform the action
                Agent update its trajectory: action taken, new state, new reward
                Agent pick an action from its new state according to its policy
                Update the stop condition if Goal reached or terminated
            '''
            observation, reward, terminated, truncated, info = environment.step(action)
            learning_agent.trajectory.append(learning_agent.current_state_index, action, float(reward))
            learning_agent.current_state_index = observation
            action = learning_agent.pick_next(observation)
            stop = terminated or truncated
        learning_agent.update()
    return Q_sa