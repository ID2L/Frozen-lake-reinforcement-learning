from __future__ import annotations # See https://stackoverflow.com/a/42845998
from typing import Any, Callable, Concatenate, Generic, ParamSpec, Protocol
import numpy as np
from abc import ABC, abstractmethod
from src.Classes.Trajectory import Trajectory
from src.Classes.Policy import Policy

# See https://github.com/microsoft/pyright/issues/3482
P = ParamSpec('P')
class Agent(Generic[P]):
    def __init__(self, policy: Policy, update_method: Callable[Concatenate[Agent, P], None] | None = None, initial_state_index: int = 0):
        self.trajectory: Trajectory = Trajectory()
        self.current_state_index = initial_state_index
        self.policy = policy
        self._update_method = update_method
        pass

    def pick_next(self, state_index:  int | None = None):
        if state_index is None:
            _state_index = self.current_state_index
        else:
            _state_index = state_index
        return self.policy.pick_next(_state_index)

    def update(self, *args: P.args, **kwargs: P.kwargs):
        if self._update_method is not None:
            return self._update_method(self, *args, **kwargs)
        return None
