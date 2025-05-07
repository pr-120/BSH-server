from abc import ABC, abstractmethod


class AbstractPreprocessor(ABC):
    @abstractmethod
    def preprocess_dataset(self, *args):
        pass
