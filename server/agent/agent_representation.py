import json
import os

from environment.state_handling import get_storage_path


class AgentRepresentation(object):
    def __init__(self, weights1, weights2, bias_weights1, bias_weights2, epsilon, learn_rate,
                 num_input, num_hidden, num_output):
        self.weights1 = weights1
        self.weights2 = weights2
        self.bias_weights1 = bias_weights1
        self.bias_weights2 = bias_weights2
        self.epsilon = epsilon
        self.learn_rate = learn_rate
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

    @staticmethod
    def save_agent(weights1, weights2, bias_weights1, bias_weights2, epsilon, agent, description):
        agent_file = os.path.join(get_storage_path(), "agent={}.json".format(description))
        content = {
            "weights1": weights1.tolist(),
            "weights2": weights2.tolist(),
            "bias_weights1": bias_weights1.tolist(),
            "bias_weights2": bias_weights2.tolist(),
            "epsilon": epsilon,
            "learn_rate": agent.learn_rate,
            "num_input": agent.num_input,
            "num_hidden": agent.num_hidden,
            "num_output": agent.num_output,
        }
        with open(agent_file, "w+") as file:
            json.dump(content, file)
        return agent_file
