from typing import Any, Callable
from nptyping import Float, NDArray, Shape
import numpy.typing as npt
import numpy as np
import random
from src.Classes.Policy import Policy


def make_epsilon_greedy_policy(epsilon: float, Q_sa: NDArray[Shape['StateDimension, ActionDimension'], Float]) -> Callable[[Policy, int], int]:
    def pickNext(policy: Policy, state: int) -> int:
        assert state >= 0 < policy.state_dimension
        rand = random.random() # pick a number in [0, 1)
        if (rand < epsilon):
            # pick uniformly between all action with probability epsilon
            return(random.randint(0, policy.action_dimension - 1))
        else:
            # pick uniformly between all action with maximum Qvalue with probability 1 - epsilon
            action_state_value = Q_sa[state, ]
            maximum_value = max(action_state_value)        
            possible_action_index = [i for i, j in enumerate(action_state_value) if j == maximum_value]
            random_between_maximum_value = random.randint(0, len(possible_action_index) - 1)
            action = possible_action_index[random_between_maximum_value]
            return action
    return pickNext 

