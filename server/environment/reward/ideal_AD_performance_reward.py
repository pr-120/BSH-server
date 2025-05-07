import math

from environment.reward.abstract_reward import AbstractReward
from environment.state_handling import collect_rate

DETECTED_ACTIONS = [1, 2]


class IdealADPerformanceReward(AbstractReward):
    def __init__(self, r_done, r_hidden, r_detected):
        self.r_done = r_done
        self.r_hidden = r_hidden
        self.r_detected = r_detected

    def compute_reward(self, action, done):
        rate = collect_rate()
        # print("REWARD: rate", rate)

        anomalous = bool(action in DETECTED_ACTIONS)  # int [0 1]
        # print("--- Detected {} FP for action {}.".format("anomalous" if anomalous else "normal", action))

        if anomalous:
            # print("REWARD: det", self.r_detected, rate, max(rate, 1))
            reward = -(max(1, abs(self.r_detected)) / max(rate, 1)) - abs(self.r_detected)  # -d/r - d
        elif done:
            reward = self.r_done
        else:
            # print("REWARD: hid", rate, 10 * math.log(rate+1), self.r_hidden)
            reward = 10 * math.log(rate + 1) + abs(self.r_hidden)  # ln(r+1) + h
        # print("REWARD: result", reward)
        return round(reward, 5), anomalous
