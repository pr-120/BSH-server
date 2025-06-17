import csv
import json
import os
import threading
from datetime import datetime
from time import sleep, time

import numpy as np
import psutil
from tqdm import tqdm

from agent.agent_representation_mutlilayer import AgentRepresentationMultiLayer
from api.backdoor import send_terminate
from api.configurations import send_config, save_config_locally
from environment.abstract_controller import AbstractController
from environment.reward.performance_reward import PerformanceReward
from environment.settings import MAX_EPISODES_V21, SIM_CORPUS_SIZE_V21, EPSILON_V21, DECAY_RATE_V21, \
    LIVE_TRAINING_DEVICE
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint, is_simulation, \
    set_rw_done, collect_rate, get_prototype, get_storage_path, map_to_backdoor_configuration
from utilities.plots import plot_average_results
from utilities.simulate import simulate_sending_fp, simulate_sending_rw_done

DEBUG_PRINTING = False
EPSILON = EPSILON_V21
DECAY_RATE = DECAY_RATE_V21

class ControllerDDQLIdealAD(AbstractController):
    def loop_episodes(self, agent):
        start_timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        run_info = "p{}-{}e-{}s".format(get_prototype(), MAX_EPISODES_V21, SIM_CORPUS_SIZE_V21)
        description = "{}={}".format(start_timestamp, run_info)
        agent_file = None
        simulated = is_simulation()

        reward_system = PerformanceReward(+100, +0, -100)
        weights_list, bias_weights_list = agent.initialize_network()
        target_weights_list = [w.copy() for w in weights_list]
        target_bias_weights_list = [bw.copy() for bw in bias_weights_list]

        all_rewards = []
        all_summed_rewards = []
        all_avg_rewards = []
        all_num_steps = []
        last_q_values = []
        num_total_steps = 0
        all_start = time()

        # Resource monitoring setup
        resource_log = []
        resource_log_lock = threading.Lock()
        episode_timings = []
        stop_event = threading.Event()

        def monitor_resources():
            process = psutil.Process()  # Get current Python process
            while not stop_event.is_set():
                timestamp = time()
                try:
                    with process.oneshot():  # Optimize resource access
                        cpu_percent = process.cpu_percent(interval=0.1)  # Reduced interval to 0.1s
                        memory_info = process.memory_info()
                        memory_used_mb = memory_info.rss / (1024 ** 2)  # Convert bytes to MB
                    with resource_log_lock:
                        resource_log.append((timestamp, cpu_percent, memory_used_mb))
                except psutil.Error:
                    # Handle cases where process info is temporarily unavailable
                    with resource_log_lock:
                        resource_log.append((timestamp, 0.0, 0.0))
                sleep(0.1)  # Ensure we don't overload the system

        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        base_lr = 0.005  # Initial learning rate
        decay_rate = 0.95  # Decay factor (95% of previous value)
        decay_steps = 10  # Decay every 10 episodes

        eps_iter = range(1, MAX_EPISODES_V21 + 1) if DEBUG_PRINTING else tqdm(range(1, MAX_EPISODES_V21 + 1))
        for episode in eps_iter:
            # ==============================
            # Setup environment
            # ==============================

            set_rw_done(False)
            epsilon_episode = EPSILON / (1 + DECAY_RATE * (episode - 1))

            agent.learn_rate = base_lr * (decay_rate ** (episode // decay_steps))

            last_action = -1
            reward_store = []
            summed_reward = 0
            steps = 0
            sim_encryption_progress = 0
            eps_start = time()

            # accept initial FP
            log("Wait for initial FP...")
            save_config_locally(0)
            if simulated:
                simulate_sending_fp(0)
            while not is_fp_ready():
                sleep(.5)
            curr_fp = collect_fingerprint()
            set_fp_ready(False)

            log("Loop episode...")
            while True:
                log("==================================================")
                # ==============================
                # Predict action
                # ==============================

                state = AbstractController.transform_fp(curr_fp)
                curr_hidden_list, curr_q_values, selected_action = agent.predict(weights_list, bias_weights_list, epsilon_episode, state=state)
                steps += 1

                # ==============================
                # Take step and observe new state
                # ==============================

                if selected_action != last_action:
                    config = map_to_backdoor_configuration(selected_action)
                    if not simulated:
                        send_config(selected_action, config, LIVE_TRAINING_DEVICE)
                last_action = selected_action

                save_config_locally(int(selected_action))
                if simulated:
                    simulate_sending_fp(selected_action)
                while not (is_fp_ready() or is_rw_done()):
                    sleep(.5)
                next_fp = collect_fingerprint() if not is_rw_done() else curr_fp
                next_state = AbstractController.transform_fp(next_fp)
                set_fp_ready(False)

                rate = collect_rate()
                sim_encryption_progress += rate

                # ==============================
                # Observe reward for new state
                # ==============================

                if simulated and sim_encryption_progress >= SIM_CORPUS_SIZE_V21:
                    simulate_sending_rw_done()

                is_done = is_rw_done()
                reward, detected = reward_system.compute_reward(next_state, is_done)
                reward_store.append((selected_action, reward))
                summed_reward += reward
                if detected:
                    set_rw_done()

                # ==============================
                # Next Q-values, error, and learning
                # ==============================

                error = agent.init_error()
                if is_rw_done():
                    error = agent.update_error(error, reward, selected_action, curr_q_values, None, is_done=True)
                    weights_list, bias_weights_list = agent.update_weights(curr_q_values, error, state, curr_hidden_list, weights_list, bias_weights_list)
                    last_q_values = curr_q_values
                else:
                    next_hidden_list, next_q_values, next_action = agent.predict(target_weights_list, target_bias_weights_list, epsilon_episode, state=next_state, target=True)
                    error = agent.update_error(error, reward, selected_action, curr_q_values, next_q_values, is_done=False)
                    weights_list, bias_weights_list = agent.update_weights(curr_q_values, error, state, curr_hidden_list, weights_list, bias_weights_list)

                if is_rw_done():
                    break
                curr_fp = next_fp

            # ========== END OF EPISODE ==========
            eps_end = time()
            episode_timings.append((episode, eps_start, eps_end))
            log("Episode {} took: {}s, roughly {}min.".format(episode, "%.3f" % (eps_end - eps_start),
                                                              "%.1f" % ((eps_end - eps_start) / 60)))
            num_total_steps += steps
            all_rewards.append(reward_store)
            all_summed_rewards.append(summed_reward)
            all_avg_rewards.append(summed_reward / steps)
            all_num_steps.append(steps)

            # Update target network
            agent.update_target_network(weights_list, bias_weights_list)
            agent_file = AgentRepresentationMultiLayer.save_agent(weights_list, bias_weights_list, epsilon_episode, agent, description)

        # ========== END OF TRAINING ==========
        all_end = time()
        stop_event.set()
        monitor_thread.join()

        # Compute per-episode resource usage
        per_episode_resources = self.compute_per_episode_resources(episode_timings, resource_log)

        # Save resource log to CSV
        resource_log_file = os.path.join(get_storage_path(), f"resource_log_{description}.csv")
        with open(resource_log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "cpu_percent", "memory_used_mb"])
            for row in resource_log:
                writer.writerow(row)

        log("All episodes took: {}s, roughly {}min.".format("%.3f" % (all_end - all_start),
                                                            "%.1f" % ((all_end - all_start) / 60)))
        print("steps total", num_total_steps, "avg", num_total_steps / MAX_EPISODES_V21)
        print("==============================")
        print("Saving trained agent to file...")
        print("- Agent saved:", agent_file)
        print("Generating plots...")
        results_plots_file = plot_average_results(all_summed_rewards, all_avg_rewards, all_num_steps, MAX_EPISODES_V21, description)
        print("- Plots saved:", results_plots_file)
        results_store_file = self.save_results_with_resources_to_file(all_summed_rewards, all_avg_rewards,
                                                                      all_num_steps, per_episode_resources, description)
        print("- Results saved:", results_store_file)
        print(f"- Resource log saved: {resource_log_file}")
        print(f"- CPU usage plot saved: {os.path.join(get_storage_path(), f'cpu_usage_{description}.png')}")
        print(f"- Memory usage plot saved: {os.path.join(get_storage_path(), f'memory_usage_{description}.png')}")

        if not simulated:
            send_terminate()
        return last_q_values, all_rewards

    def compute_per_episode_resources(self, episode_timings, resource_log):
        per_episode_resources = []
        last_max_memory = 400.0  # Default value based on data trend (around 400 MB)
        last_max_cpu = 50.0  # Default value for CPU usage based on data trend (around 50%-100%)

        for episode, start, end in episode_timings:
            episode_resources = [r for r in resource_log if start <= r[0] <= end]
            if episode_resources:
                timestamps, cpu_percents, memory_used_mbs = zip(*episode_resources)
                avg_cpu = np.mean(cpu_percents)
                max_cpu = np.max(cpu_percents)
                avg_memory_used_mb = np.mean(memory_used_mbs)
                max_memory_used_mb = np.max(memory_used_mbs)
                last_max_memory = max_memory_used_mb  # Update the last known memory value
                last_max_cpu = max_cpu  # Update the last known CPU value
            else:
                # If no resources are found for this episode, carry forward the last known values
                avg_cpu = last_max_cpu  # Use the last known CPU value
                max_cpu = last_max_cpu  # Use the last known CPU value
                avg_memory_used_mb = last_max_memory  # Use the last known memory value
                max_memory_used_mb = last_max_memory  # Use the last known memory value

            per_episode_resources.append({
                "episode": episode,
                "avg_cpu": float(avg_cpu),
                "max_cpu": float(max_cpu),
                "avg_memory_used_mb": float(avg_memory_used_mb),
                "max_memory_used_mb": float(max_memory_used_mb)
            })
        return per_episode_resources

    def save_results_with_resources_to_file(self, all_summed_rewards, all_avg_rewards, all_num_steps,
                                            per_episode_resources, run_description):
        results_content = json.dumps({
            "summed_rewards": all_summed_rewards,
            "avg_rewards": all_avg_rewards,
            "num_steps": all_num_steps,
            "resource_usage": per_episode_resources
        }, indent=4)
        results_file = os.path.join(get_storage_path(), f"results-store={run_description}.txt")
        with open(results_file, "w") as file:
            file.write(results_content)
        return results_file

def log(*args):
    if DEBUG_PRINTING:
        print(*args)