import math

from environment.anomaly_detection.anomaly_detection import detect_anomaly
from environment.reward.abstract_reward import AbstractReward
from environment.state_handling import collect_rate


class PerformanceReward(AbstractReward):
    def __init__(self, r_done, r_hidden, r_detected):
        self.r_done = r_done
        self.r_hidden = r_hidden
        self.r_detected = r_detected

    def compute_reward(self, fp, done):
        rate = collect_rate()
        # print("REWARD: rate", rate)

        # One-Class SVM returns -1 for anomalous and 1 for normal
        anomalous = detect_anomaly(fp)  # int [0 1]
        if anomalous == 1:
            anomalous = False
        else:
            anomalous = True

        # print("--- Detected {} FP.".format("anomalous" if anomalous else "normal"))

        if anomalous:
            # print("REWARD: det", self.r_detected, rate, max(rate, 1))
            reward = -(max(1, abs(self.r_detected)) / max(rate, 1)) - abs(self.r_detected)  # -d/r - d
        elif done:
            reward = self.r_done
        else:
            # print("REWARD: hid", rate, 10 * math.log(rate+1), self.r_hidden)
            reward = 100 * math.log(rate / 100 + 1) + abs(self.r_hidden)  # ln(r+1) + h
        # print("REWARD: result", reward)
        return round(reward, 5), anomalous
