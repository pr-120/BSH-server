import json
import os
from abc import ABC, abstractmethod
from time import sleep

import numpy as np

from agent.agent_representation import AgentRepresentation
from agent.constructor import get_agent, build_agent_from_repr
from environment.reward.abstract_reward import AbstractReward
from environment.state_handling import is_api_running, is_simulation, get_prototype, get_storage_path,\
    get_agent_representation_path

WAIT_FOR_CONFIRM = False


class AbstractController(ABC):
    @abstractmethod
    def loop_episodes(self, agent):
        pass

    def __start_training(self):
        if not is_simulation():
            print("\nWaiting for API...")
            while not is_api_running():
                sleep(1)

        print("\n==============================\nStart Training\n==============================")
        # Initialize random seed for reproducibility
        np.random.seed(42)

        representation_path = get_agent_representation_path()
        if isinstance(representation_path, str) and len(representation_path) > 0\
                and os.path.exists(representation_path):
            print("Verified agent representation path:", representation_path)
            with open(representation_path, "r") as agent_file:
                repr_dict = json.load(agent_file)
            representation = AgentRepresentation(repr_dict["weights1"], repr_dict["weights2"],
                                                 repr_dict["bias_weights1"], repr_dict["bias_weights2"],
                                                 repr_dict["epsilon"], repr_dict["learn_rate"],
                                                 repr_dict["num_input"], repr_dict["num_hidden"],
                                                 repr_dict["num_output"])
            print("Building agent from representation.")
            agent = build_agent_from_repr(representation)
            print("------------------------------")
        else:
            agent = get_agent()

        q_values, rewards = self.loop_episodes(agent)
        if int(get_prototype()) > 1:
            print("==============================", "Rewards:", rewards, "\nFinal Q-Values:", q_values, sep="\n")
        print("\n==============================\n! Done !\n==============================")

    def run_c2(self):
        print("==============================\nPrepare Reward Computation\n==============================")
        AbstractReward.prepare_reward_computation()

        if WAIT_FOR_CONFIRM:
            cont = input("Results ok? Start C2 Server? [y/n]\n")
            if cont.lower() == "y":
                self.__start_training()
        else:
            self.__start_training()

    @staticmethod
    def transform_fp(fp):
        split_to_floats = list(map(lambda feat: float(feat), fp.split(",")))
        return np.asarray(split_to_floats).reshape(-1, 1)  # shape (F, 1)

    @staticmethod
    def save_results_to_file(all_summed_rewards, all_avg_rewards, all_num_steps, run_description):
        results_content = "\n".join(["{",
                                     '"summed_rewards": {},'.format(all_summed_rewards),
                                     '"avg_rewards": {},'.format(all_avg_rewards),
                                     '"num_steps": {}'.format(all_num_steps),
                                     "}\n"])
        results_file = os.path.join(get_storage_path(), "results-store={}.txt".format(run_description))
        with open(results_file, "w") as file:
            file.write(results_content)
        return results_file
