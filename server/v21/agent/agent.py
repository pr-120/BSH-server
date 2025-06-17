import math
import os
import numpy as np
import pandas as pd
from agent.abstract_agent import AbstractAgent
from agent.agent_representation_mutlilayer import AgentRepresentationMultiLayer
from environment.anomaly_detection.constructor import get_preprocessor
from environment.settings import ALL_CSV_HEADERS, TRAINING_CSV_FOLDER_PATH, LEARN_RATE_V21, DISCOUNT_FACTOR_V21
from environment.state_handling import get_num_configs
from v21.agent.model import ModelOptimized

LEARN_RATE = LEARN_RATE_V21
DISCOUNT_FACTOR = DISCOUNT_FACTOR_V21

class AgentDDQLIdealAD(AbstractAgent):
    def __init__(self, representation=None, hidden_sizes=None):
        self.representation = representation
        # Default to 2 hidden layers if not specified
        if hidden_sizes is None:
            num_configs = get_num_configs()
            self.num_input = len(self.__get_fp_features())
            hidden_sizes = [round(self.num_input * 2), round(self.num_input)]
        self.hidden_sizes = hidden_sizes

        if isinstance(representation, AgentRepresentationMultiLayer):
            self.num_input = representation.num_input
            self.num_output = representation.num_output
            self.actions = list(range(self.num_output))
            self.learn_rate = representation.learn_rate
            self.fp_features = self.__get_fp_features()
            self.model = ModelOptimized(learn_rate=self.learn_rate, num_configs=self.num_output, hidden_sizes=self.hidden_sizes)
            self.target_model = ModelOptimized(learn_rate=self.learn_rate, num_configs=self.num_output, hidden_sizes=self.hidden_sizes)
        else:
            num_configs = get_num_configs()
            self.actions = list(range(num_configs))
            self.fp_features = self.__get_fp_features()
            self.num_input = len(self.fp_features)
            self.num_output = num_configs
            self.learn_rate = LEARN_RATE
            self.model = ModelOptimized(learn_rate=self.learn_rate, num_configs=num_configs, hidden_sizes=self.hidden_sizes)
            self.target_model = ModelOptimized(learn_rate=self.learn_rate, num_configs=num_configs, hidden_sizes=self.hidden_sizes)

    def __get_fp_features(self):
        df_normal = pd.read_csv(os.path.join(TRAINING_CSV_FOLDER_PATH, "normal-behavior.csv"))
        preprocessor = get_preprocessor()
        ready_dataset = preprocessor.preprocess_dataset(df_normal)
        return ready_dataset.columns

    def __preprocess_fp(self, fp):
        headers = ALL_CSV_HEADERS.split(",")
        indexes = [headers.index(header) for header in self.fp_features]
        return fp[indexes]

    def initialize_network(self):
        if isinstance(self.representation, AgentRepresentationMultiLayer):
            weights_list = [np.asarray(w) for w in self.representation.weights_list]
            bias_weights_list = [np.asarray(bw) for bw in self.representation.bias_weights_list]
        else:
            # He weight initialization for multiple layers
            layer_sizes = [self.num_input] + self.hidden_sizes + [self.num_output]
            weights_list = []
            bias_weights_list = []
            for i in range(len(layer_sizes) - 1):
                n_prev = layer_sizes[i]
                n_next = layer_sizes[i + 1]
                std = math.sqrt(2.0 / n_prev)
                weights = np.random.randn(n_prev, n_next) * std
                bias_weights = np.zeros((n_next, 1))
                weights_list.append(weights)
                bias_weights_list.append(bias_weights)
        return weights_list, bias_weights_list

    def predict(self, weights_list, bias_weights_list, epsilon, state, target=False):
        std_fp = AbstractAgent.standardize_fp(state)
        ready_fp = self.__preprocess_fp(std_fp)
        if target:
            hidden_list, q_values, selected_action = self.target_model.forward(weights_list, bias_weights_list, epsilon, inputs=ready_fp)
        else:
            hidden_list, q_values, selected_action = self.model.forward(weights_list, bias_weights_list, epsilon, inputs=ready_fp)
        return hidden_list, q_values, selected_action

    def update_weights(self, q_values, error, state, hidden_list, weights_list, bias_weights_list):
        std_fp = AbstractAgent.standardize_fp(state)
        ready_fp = self.__preprocess_fp(std_fp)
        new_weights_list, new_bias_weights_list = self.model.backward(q_values, error, hidden_list, weights_list, bias_weights_list, inputs=ready_fp, learn_rate=self.learn_rate)
        return new_weights_list, new_bias_weights_list

    def init_error(self):
        return np.zeros((self.num_output, 1))

    def update_error(self, error, reward, selected_action, curr_q_values, next_q_values, is_done):
        if is_done:
            error[selected_action] = reward - curr_q_values[selected_action]
        else:
            error[selected_action] = reward + (DISCOUNT_FACTOR * next_q_values[np.argmax(curr_q_values)]) - curr_q_values[selected_action]
        return error

    def update_target_network(self, weights_list, bias_weights_list):
        self.target_model.weights_list = [w.copy() for w in weights_list]
        self.target_model.bias_weights_list = [bw.copy() for bw in bias_weights_list]