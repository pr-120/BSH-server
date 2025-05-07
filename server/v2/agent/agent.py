import os

import numpy as np

from agent.abstract_agent import AbstractAgent
from agent.agent_representation import AgentRepresentation
from environment.settings import TRAINING_CSV_FOLDER_PATH, ALL_CSV_HEADERS, DUPLICATE_HEADERS
from environment.state_handling import get_num_configs
from v2.agent.model import ModelQLearning

USE_SIMPLE_FP = False
FP_DIMS = 7 if USE_SIMPLE_FP else 97
HIDDEN_NEURONS = 10 if USE_SIMPLE_FP else 50

LEARN_RATE = 0.05 if USE_SIMPLE_FP else 0.005
DISCOUNT_FACTOR = 0.5 if USE_SIMPLE_FP else 0.75


class AgentQLearning(AbstractAgent):
    def __init__(self, representation=None):
        self.representation = representation

        if isinstance(representation, AgentRepresentation):  # build from representation
            self.num_input = representation.num_input
            self.num_hidden = representation.num_hidden
            self.num_output = representation.num_output
            self.actions = list(range(representation.num_output))

            self.learn_rate = representation.learn_rate
            self.model = ModelQLearning(learn_rate=LEARN_RATE, num_configs=self.num_output)
        else:  # init from scratch
            num_configs = get_num_configs()
            self.num_input = FP_DIMS  # Input size
            self.num_hidden = HIDDEN_NEURONS  # Hidden neurons
            self.num_output = num_configs  # Output size
            self.actions = list(range(num_configs))

            self.learn_rate = LEARN_RATE  # only used in AbstractAgent for storing AgentRepresentation
            self.model = ModelQLearning(learn_rate=LEARN_RATE, num_configs=num_configs)

    def __preprocess_fp(self, fp):
        headers = ALL_CSV_HEADERS.split(",")

        duplicates = set(DUPLICATE_HEADERS)  # 3 features
        duplicates_included = []

        # only time metrics and duplicates
        # 3+3 features dropped, leaves 103 - 6 = 97 features
        dropped_features = ["time", "timestamp", "seconds"]

        indexes = []
        for header, value in zip(headers, fp):
            if header not in dropped_features:
                if header not in duplicates:
                    indexes.append(headers.index(header))
                else:
                    if header not in duplicates_included:
                        indexes.append(headers.index(header))
                        duplicates_included.append(header)

        return fp[indexes]

    def __crop_fp(self, fp):
        with open(os.path.join(TRAINING_CSV_FOLDER_PATH, "normal-behavior.csv"), "r") as csv_normal:
            csv_headers = csv_normal.read().split(",")
        headers = ["cpu_id", "tasks_running", "mem_free", "cpu_temp", "block:block_bio_remap",
                   "sched:sched_process_exec", "writeback:writeback_pages_written"]
        indexes = []
        for header in headers:
            indexes.append(csv_headers.index(header))
        return fp[indexes]

    def initialize_network(self):
        if isinstance(self.representation, AgentRepresentation):  # init from representation
            weights1 = np.asarray(self.representation.weights1)
            weights2 = np.asarray(self.representation.weights2)
            bias_weights1 = np.asarray(self.representation.bias_weights1)
            bias_weights2 = np.asarray(self.representation.bias_weights2)
        else:  # init from scratch
            # uniform weight initialization
            weights1 = np.random.uniform(0, 1, (self.num_input, self.num_hidden))
            weights2 = np.random.uniform(0, 1, (self.num_hidden, self.num_output))

            bias_weights1 = np.zeros((self.num_hidden, 1))
            bias_weights2 = np.zeros((self.num_output, 1))

        return weights1, weights2, bias_weights1, bias_weights2

    def predict(self, weights1, weights2, bias_weights1, bias_weights2, epsilon, state):
        std_fp = AbstractAgent.standardize_fp(state)
        if USE_SIMPLE_FP:
            ready_fp = self.__crop_fp(std_fp)
        else:
            ready_fp = self.__preprocess_fp(std_fp)
        hidden, q_values, selected_action = self.model.forward(weights1, weights2, bias_weights1, bias_weights2,
                                                               epsilon, inputs=ready_fp)
        return hidden, q_values, selected_action

    def update_weights(self, q_values, error, state, hidden, weights1, weights2, bias_weights1, bias_weights2):
        std_fp = AbstractAgent.standardize_fp(state)
        if USE_SIMPLE_FP:
            ready_fp = self.__crop_fp(std_fp)
        else:
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
