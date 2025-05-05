from abc import ABC, abstractmethod

from environment.anomaly_detection.anomaly_detection import train_anomaly_detection


class AbstractReward(ABC):
    @staticmethod
    def prepare_reward_computation():
        train_anomaly_detection()

    @abstractmethod
    def compute_reward(self, *args):
        pass
