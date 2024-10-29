import gymnasium as gym
import numpy as np


class BudgetActionWrapper(gym.Wrapper):

    def __init__(self, env, budget=5) -> None:
        super(BudgetActionWrapper, self).__init__(env)
        self._last_action = np.full_like(self.action_space.low, -np.inf)
        self._budget = budget+1 # first action also takes from budget
        self._action_margin = 0.05

    def augment_state(self, state):
        if type(state) == tuple:
            state = (self._budget_left.copy(), *state)
        else:
            state = (self._budget_left.copy(), state)
        return state

    def reset(self):
        self._budget_left = np.full_like(self.action_space.low, self._budget)
        self._last_action = np.full_like(self.action_space.low, -np.inf)
        s = super(BudgetActionWrapper, self).reset()
        return self.augment_state(s)

    def step(self, raw_action):
        action = self._last_action.copy()
        change_point = np.abs(self._last_action - raw_action) > self._action_margin
        no_change_point = np.logical_not(change_point)
        action[no_change_point] = raw_action[no_change_point]
        change_point_budget = np.logical_and(change_point, self._budget_left > 0)
        action[change_point_budget] = raw_action[change_point_budget]
        # decrease budget
        self._budget_left[change_point_budget] -= 1
        # update last action
        self._last_action = action

        ns, r, d, info = super(BudgetActionWrapper, self).step(action)
        info['action'] = action.copy()

        # augment next state
        ns = self.augment_state(ns)
        return ns, r, d, info

    