import numpy as np

from agent.abstract_agent import AbstractAgent
from environment.settings import MAX_STEPS_V99
from environment.state_handling import get_num_configs


class AgentBruteForce(AbstractAgent):
    def __init__(self):
        # counters
        self.action = 0
        self.step = 1
        # max constants
        self.num_actions = get_num_configs()
        self.max_steps = MAX_STEPS_V99

        self.weights = np.zeros(self.num_actions)

    def predict(self, fingerprint):
        print("AGENT: action-step", self.action, self.step)
        # take action N=step times
        next_a = self.action
        self.step += 1

        # take next action and reset steps
        if self.step > self.max_steps:
            self.action += 1
            self.step = 1

        # after the last step of the last action, we increase action over num_actions
        is_last = self.action >= self.num_actions
        return next_a, is_last

    def update_weights(self, _, reward):
        if self.action >= self.num_actions:  # last action was chosen
            idx = self.num_actions - 1
        elif self.step == 1:  # action just changed
            idx = self.action - 1
        else:
            idx = self.action

        self.weights[idx] += reward
        return self.weights
