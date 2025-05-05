from datetime import datetime
from time import sleep, time

from tqdm import tqdm  # add progress bar to episodes

from agent.agent_representation import AgentRepresentation
from api.configurations import map_to_ransomware_configuration, send_config
from environment.abstract_controller import AbstractController
from environment.reward.performance_reward import PerformanceReward
from environment.settings import MAX_EPISODES_V6, SIM_CORPUS_SIZE_V6
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint, is_simulation, \
    set_rw_done, collect_rate, get_prototype
from utilities.plots import plot_average_results
from utilities.simulate import simulate_sending_fp, simulate_sending_rw_done

DEBUG_PRINTING = False

EPSILON = 0.5
DECAY_RATE = 0.01


class ControllerSarsa(AbstractController):
    def loop_episodes(self, agent):
        start_timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        run_info = "p{}-{}e-{}s".format(get_prototype(), MAX_EPISODES_V6, SIM_CORPUS_SIZE_V6)
        description = "{}={}".format(start_timestamp, run_info)
        agent_file = None

        reward_system = PerformanceReward(+1000, +0, -20)
        weights1, weights2, bias_weights1, bias_weights2 = agent.initialize_network()

        # ==============================
        # Setup collectibles
        # ==============================

        all_rewards = []
        all_summed_rewards = []
        all_avg_rewards = []
        all_num_steps = []

        last_q_values = []
        num_total_steps = 0
        all_start = time()

        eps_iter = range(1, MAX_EPISODES_V6 + 1) if DEBUG_PRINTING else tqdm(range(1, MAX_EPISODES_V6 + 1))
        for episode in eps_iter:
            # ==============================
            # Setup environment
            # ==============================

            set_rw_done(False)

            epsilon_episode = EPSILON / (1 + DECAY_RATE * (episode - 1))  # decay epsilon, episode 1-based

            last_action = -1
            reward_store = []
            summed_reward = 0

            steps = 1
            sim_encryption_progress = 0
            eps_start = time()

            # accept initial FP
            log("Wait for initial FP...")
            if is_simulation():
                simulate_sending_fp(0)
            while not is_fp_ready():
                sleep(.5)
            curr_fp = collect_fingerprint()
            set_fp_ready(False)

            # ==============================
            # Episodes
            # ==============================

            # transform FP into np array
            state = AbstractController.transform_fp(curr_fp)

            # agent selects action based on state
            log("Predict initial action.")
            curr_hidden, curr_q_values, selected_action = agent.predict(weights1, weights2, bias_weights1,
                                                                        bias_weights2, epsilon_episode, state=state)
            log("Predicted action {}. Episode {} step {}.".format(selected_action, episode, steps))

            log("Loop episode...")
            while not is_rw_done():
                log("==================================================")
                # ==============================
                # Take step and observe new state
                # ==============================

                # convert action to config and send to client
                if selected_action != last_action:
                    log("Sending new action {} to client.".format(selected_action))
                    config = map_to_ransomware_configuration(selected_action)
                    if not is_simulation():  # cannot send if no socket listening during simulation
                        send_config(selected_action, config)
                last_action = selected_action

                # receive next FP and compute reward based on FP
                log("Wait for FP...")
                if is_simulation():
                    simulate_sending_fp(selected_action)
                while not (is_fp_ready() or is_rw_done()):
                    sleep(.5)

                if is_rw_done():
                    next_fp = curr_fp
                else:
                    next_fp = collect_fingerprint()

                next_state = AbstractController.transform_fp(next_fp)
                set_fp_ready(False)

                # compute encryption progress (assume 1s per step) and reported encryption rate for simulation
                rate = collect_rate()
                sim_encryption_progress += rate

                # ==============================
                # Observe reward for new state
                # ==============================

                if is_simulation() and sim_encryption_progress >= SIM_CORPUS_SIZE_V6:
                    simulate_sending_rw_done()

                log("Computing reward for next FP.")
                reward, detected = reward_system.compute_reward(next_state, is_rw_done())
                log("Computed reward", reward)
                reward_store.append((selected_action, reward))
                summed_reward += reward
                if detected:
                    set_rw_done()  # terminate episode

                # ==============================
                # Next Q-values, error, and learning
                # ==============================

                # initialize error
                error = agent.init_error()

                if is_rw_done():
                    # update error based on observed reward
                    error = agent.update_error(error, reward, selected_action, curr_q_values[selected_action],
                                               next_action=None, next_q_value=None, is_done=True)

                    # send error to agent, update weights accordingly
                    weights1, weights2, bias_weights1, bias_weights2 = agent.update_weights(curr_q_values, error, state,
                                                                                            curr_hidden, weights1,
                                                                                            weights2, bias_weights1,
                                                                                            bias_weights2)
                    last_q_values = curr_q_values
                else:
                    # predict next Q-values and action
                    log("Predict next action.")
                    next_hidden, next_q_values, next_selected_action = agent.predict(weights1, weights2, bias_weights1,
                                                                                     bias_weights2, epsilon_episode,
                                                                                     state=next_state)
                    steps += 1
                    log("Predicted next action {}. Episode {} step {}.".format(next_selected_action, episode, steps))

                    # update error based on observed reward
                    error = agent.update_error(error, reward, selected_action, curr_q_values[selected_action],
                                               next_selected_action, next_q_values[next_selected_action], is_done=False)

                    # send error to agent, update weights accordingly
                    weights1, weights2, bias_weights1, bias_weights2 = agent.update_weights(curr_q_values, error, state,
                                                                                            curr_hidden, weights1,
                                                                                            weights2, bias_weights1,
                                                                                            bias_weights2)

                    # ==============================
                    # Prepare next step
                    # ==============================

                    # update current state
                    curr_fp = next_fp
                    selected_action = next_selected_action
                # ========== END OF STEP ==========

            # ========== END OF EPISODE ==========
            eps_end = time()
            log("Episode {} took: {}s, roughly {}min.".format(episode, "%.3f" % (eps_end - eps_start),
                                                              "%.1f" % ((eps_end - eps_start) / 60)))
            # print("Episode {} had {} steps.".format(episode, steps))
            num_total_steps += steps
            all_rewards.append(reward_store)
            all_summed_rewards.append(summed_reward)
            all_avg_rewards.append(summed_reward / steps)  # average reward over episode
            all_num_steps.append(steps)

            agent_file = AgentRepresentation.save_agent(weights1, weights2, bias_weights1, bias_weights2,
                                                        epsilon_episode, agent, description)
            log("=================================================\n=================================================")

        # ========== END OF TRAINING ==========
        all_end = time()
        log("All episodes took: {}s, roughly {}min.".format("%.3f" % (all_end - all_start),
                                                            "%.1f" % ((all_end - all_start) / 60)))
        print("steps total", num_total_steps, "avg", num_total_steps / MAX_EPISODES_V6)

        print("==============================")
        print("Saving trained agent to file...")
        print("- Agent saved:", agent_file)

        print("Generating plots...")
        results_plots_file = plot_average_results(all_summed_rewards, all_avg_rewards, all_num_steps, MAX_EPISODES_V6,
                                                  description)
        print("- Plots saved:", results_plots_file)
        results_store_file = AbstractController.save_results_to_file(all_summed_rewards, all_avg_rewards, all_num_steps,
                                                                     description)
        print("- Results saved:", results_store_file)
        return last_q_values, all_rewards


def log(*args):
    if DEBUG_PRINTING:  # tqdm replaces progress inline, so prints would spam the console with multiple progress bars
        print(*args)
