from time import sleep

from api.configurations import map_to_ransomware_configuration, send_config
from environment.abstract_controller import AbstractController
from environment.reward.standard_reward import StandardReward
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint, is_simulation
from utilities.simulate import simulate_sending_fp


class ControllerManual(AbstractController):
    def loop_episodes(self, agent):
        # setup
        reward_system = StandardReward(+1, 0, -1)
        last_action = None

        # accept initial FP
        print("Wait for initial FP...")
        if is_simulation():
            simulate_sending_fp(0)
        while not is_fp_ready():
            sleep(.5)
        curr_fp = collect_fingerprint()
        set_fp_ready(False)

        # print("Loop episode...")
        while True:
            # transform FP into np array
            state = AbstractController.transform_fp(curr_fp)

            # agent selects action based on state
            print("Predict next action.")
            selected_action, is_last = agent.predict(state)
            print("Predicted action {}; is last: {}.".format(selected_action, is_last))

            # convert action to config and send to client
            if selected_action != last_action:
                print("Sending new action {} to client.".format(selected_action))
                config = map_to_ransomware_configuration(selected_action)
                if not is_simulation():  # cannot send if no socket listening during simulation
                    send_config(selected_action, config)
            last_action = selected_action

            # receive next FP and compute reward based on FP
            print("Wait for FP...")
            if is_simulation():
                simulate_sending_fp(selected_action)
            while not (is_fp_ready()):
                sleep(.5)

            next_fp = collect_fingerprint()
            set_fp_ready(False)

            print("Computing reward for next FP.")
            reward = reward_system.compute_reward(AbstractController.transform_fp(next_fp), is_rw_done())

            if is_last:
                # terminate episode instantly
                print("Terminate episode.")
                break
            # set next_fp to curr_fp for next iteration
            curr_fp = next_fp

        return [], []
