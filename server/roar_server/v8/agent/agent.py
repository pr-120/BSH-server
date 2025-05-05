import math
import os

import numpy as np
import pandas as pd

from agent.abstract_agent import AbstractAgent
from agent.agent_representation import AgentRepresentation
from environment.anomaly_detection.constructor import get_preprocessor
from environment.settings import ALL_CSV_HEADERS, TRAINING_CSV_FOLDER_PATH
from environment.state_handling import get_num_configs
from v8.agent.model import ModelOptimized

LEARN_RATE = 0.0050
DISCOUNT_FACTOR = 0.10


class AgentOptimized(AbstractAgent):
    def __init__(self, representation=None):
        self.representation = representation
        if isinstance(representation, AgentRepresentation):  # build from representation
            self.num_input = representation.num_input
            self.num_hidden = representation.num_hidden
            self.num_output = representation.num_output
            self.actions = list(range(representation.num_output))

            self.learn_rate = representation.learn_rate
            self.fp_features = self.__get_fp_features()
            self.model = ModelOptimized(learn_rate=self.learn_rate, num_configs=self.num_output)
        else:  # init from scratch
            num_configs = get_num_configs()
            self.actions = list(range(num_configs))

            self.fp_features = self.__get_fp_features()

            self.num_input = len(self.fp_features)  # Input size
            self.num_hidden = round(self.num_input * 0.8 / 5) * 5  # Hidden neurons, round 80 % to 5
            self.num_output = num_configs  # Output size

            self.learn_rate = LEARN_RATE
            self.model = ModelOptimized(learn_rate=self.learn_rate, num_configs=num_configs)

    def __get_fp_features(self):
        df_normal = pd.read_csv(os.path.join(TRAINING_CSV_FOLDER_PATH, "normal-behavior.csv"))
        preprocessor = get_preprocessor()
        ready_dataset = preprocessor.preprocess_dataset(df_normal)
        return ready_dataset.columns

    def __preprocess_fp(self, fp):
        headers = ALL_CSV_HEADERS.split(",")
        indexes = []
        for header in self.fp_features:
            indexes.append(headers.index(header))
        return fp[indexes]

    def initialize_network(self):
        if isinstance(self.representation, AgentRepresentation):  # init from representation
            weights1 = np.asarray(self.representation.weights1)
            weights2 = np.asarray(self.representation.weights2)
            bias_weights1 = np.asarray(self.representation.bias_weights1)
            bias_weights2 = np.asarray(self.representation.bias_weights2)
        else:  # init from scratch
            # He weight initialization
            # number of nodes in the previous layer
            n1 = self.num_input
            n2 = self.num_hidden
            n3 = self.num_output
            # calculate the range for the weights
            std1 = math.sqrt(2.0 / n1)
            std2 = math.sqrt(2.0 / n2)
            # generate random numbers
            numbers1 = np.random.randn(n1, n2)
            numbers2 = np.random.randn(n2, n3)
            # scale to the desired range
            weights1 = numbers1 * std1
            weights2 = numbers2 * std2

            bias_weights1 = np.zeros((self.num_hidden, 1))
            bias_weights2 = np.zeros((self.num_output, 1))

        return weights1, weights2, bias_weights1, bias_weights2

    def predict(self, weights1, weights2, bias_weights1, bias_weights2, epsilon, state):
        std_fp = AbstractAgent.standardize_fp(state)
        ready_fp = self.__preprocess_fp(std_fp)
        hidden, q_values, selected_action = self.model.forward(weights1, weights2, bias_weights1, bias_weights2,
                                                               epsilon, inputs=ready_fp)
        return hidden, q_values, selected_action

    def update_weights(self, q_values, error, state, hidden, weights1, weights2, bias_weights1, bias_weights2):
        std_fp = AbstractAgent.standardize_fp(state)
        ready_fp = self.__preprocess_fp(std_fp)
        new_w1, new_w2, new_bw1, new_bw2 = self.model.backward(q_values, error, hidden, weights1, weights2,
                                                               bias_weights1, bias_weights2, inputs=ready_fp)
        return new_w1, new_w2, new_bw1, new_bw2

    def init_error(self):
        return np.zeros((self.num_output, 1))

    def update_error(self, error, reward, selected_action, curr_q_values, next_q_values, is_done):
        # print("AGENT: R sel selval best bestval", reward, selected_action, curr_q_values, next_q_values)
        if is_done:
            error[selected_action] = reward - curr_q_values[selected_action]
        else:
            # off-policy
            error[selected_action] = reward + (DISCOUNT_FACTOR * np.max(next_q_values)) - curr_q_values[selected_action]
        # print("AGENT: err\n", error.T)
        return error
