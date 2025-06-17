import json
import os
from environment.state_handling import get_storage_path

class AgentRepresentationMultiLayer(object):
    def __init__(self, weights_list, bias_weights_list, epsilon, learn_rate, num_input, num_output):
        self.weights_list = weights_list  # List of weight matrices
        self.bias_weights_list = bias_weights_list  # List of bias vectors
        self.epsilon = epsilon
        self.learn_rate = learn_rate
        self.num_input = num_input
        self.num_output = num_output

    @staticmethod
    def save_agent(weights_list, bias_weights_list, epsilon, agent, description):
        agent_file = os.path.join(get_storage_path(), "agent={}.json".format(description))
        content = {
            "weights_list": [w.tolist() for w in weights_list],
            "bias_weights_list": [bw.tolist() for bw in bias_weights_list],
            "epsilon": epsilon,
            "learn_rate": agent.learn_rate,
            "num_input": agent.num_input,
            "num_output": agent.num_output,
        }
        with open(agent_file, "w+") as file:
            json.dump(content, file)
        return agent_file