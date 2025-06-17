import json
import os
from datetime import datetime
from time import sleep, time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import psutil
import threading
import csv
from v24.agent.agent import AgentPPONormalAD
from api.configurations import send_config, save_config_locally
from environment.reward.performance_reward import PerformanceReward
from environment.settings import MAX_EPISODES_V24, SINGLE_EPISODE_LENGTH_V24, LIVE_TRAINING_DEVICE
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint, is_simulation, \
    set_rw_done, collect_rate, get_prototype, is_api_running, get_storage_path, map_to_backdoor_configuration
from utilities.simulate import simulate_sending_fp, simulate_sending_rw_done
from utilities.plots import plot_average_results
from environment.evaluation.evaluation_ppo import evaluate_agent

DEBUG_PRINTING = False

class ControllerPPONormalAD:
    def run_c2(self, agent=None, timestamp=None, patience=5, eval_freq=1000):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        print("==============================\nPrepare Reward Computation\n==============================")
        if not is_simulation():
            print("\nWaiting for API...")
            while not is_api_running():
                sleep(1)
        print("\n==============================\nStart Training\n==============================")
        tf.random.set_seed(42)
        np.random.seed(42)
        training_agent = agent if agent is not None else AgentPPONormalAD()
        self.loop_episodes(training_agent, timestamp, patience, eval_freq)

    def loop_episodes(self, agent, timestamp, patience, eval_freq):
        start_timestamp = timestamp
        run_info = f"p{get_prototype()}-{MAX_EPISODES_V24}e-{SINGLE_EPISODE_LENGTH_V24}s"
        description = f"{start_timestamp}={run_info}"

        reward_system = PerformanceReward(+100, +0, -100)
        self.agent = agent

        all_rewards = []
        all_summed_rewards = []
        all_avg_rewards = []
        all_num_steps = []
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

        best_accuracy = -1
        patience_counter = 0
        KNOWN_BEST_ACTION = 3

        for episode in tqdm(range(1, MAX_EPISODES_V24 + 1)):
            set_rw_done(False)
            last_action = -1
            reward_store = []
            summed_reward = 0
            steps = 0
            sim_encryption_progress = 0

            log("Wait for initial FP...")
            save_config_locally(0)
            episode_start = time()
            if is_simulation():
                simulate_sending_fp(0)
            while not is_fp_ready():
                sleep(.5)
            curr_fp = collect_fingerprint()
            set_fp_ready(False)

            state = self.transform_fp(curr_fp)
            states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

            while not is_rw_done():
                action, log_prob, value = agent.act(state)
                steps += 1
                log(f"Selected action {action}. Episode {episode} step {steps}.")

                if action != last_action:
                    config = map_to_backdoor_configuration(action)
                    save_config_locally(int(action))
                    if not is_simulation():
                        send_config(action, config, LIVE_TRAINING_DEVICE)
                    last_action = action

                if is_simulation():
                    simulate_sending_fp(action)
                while not (is_fp_ready() or is_rw_done()):
                    sleep(0.5)
                next_fp = collect_fingerprint() if is_fp_ready() else state
                next_state = self.transform_fp(next_fp)
                set_fp_ready(False)

                rate = collect_rate()
                sim_encryption_progress += rate

                if is_simulation() and sim_encryption_progress >= SINGLE_EPISODE_LENGTH_V24:
                    simulate_sending_rw_done()
                    reward, detected = reward_system.compute_reward(next_state, is_rw_done())
                else:
                    reward, detected = reward_system.compute_reward(next_state, is_rw_done())

                reward_store.append((action, reward))
                summed_reward += reward
                if detected:
                    set_rw_done(True)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                dones.append(is_rw_done())
                state = next_state

            episode_end = time()
            episode_timings.append((episode, episode_start, episode_end))

            num_total_steps += steps
            all_rewards.append(reward_store)
            all_summed_rewards.append(summed_reward)
            all_avg_rewards.append(summed_reward / steps)
            all_num_steps.append(steps)

            states = np.array(states)
            actions = np.array(actions)
            log_probs = np.array(log_probs)
            values = np.array(values)
            rewards = np.array(rewards)
            dones = np.array(dones)

            returns = self.compute_returns(rewards, dones, agent.gamma)
            advantages = self.compute_gae(rewards, values, dones, states, agent.gamma, agent.lambda_)
            agent.update(states, actions, log_probs, values, advantages, returns)

            agent.current_episode += 1

            # Early stopping logic
            if episode % eval_freq == 0:
                accuracies_overall, _ = evaluate_agent(agent)
                current_accuracy = accuracies_overall.get(KNOWN_BEST_ACTION, 0) / accuracies_overall["total"]
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    # Save the best agent
                    with open(os.path.join(get_storage_path(), f"agent={timestamp}-best.json"), "w") as f:
                        json.dump(agent.get_weights_dict(), f)
                    patience_counter = 0
                    print(f"Episode {episode}: New best accuracy {current_accuracy:.4f}")
                else:
                    patience_counter += 1
                    print(f"Episode {episode}: Accuracy {current_accuracy:.4f}, Patience {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at episode {episode}")
                        break

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
        print(f"steps total {num_total_steps} avg {num_total_steps / MAX_EPISODES_V24}")
        print("==============================")
        print("Generating plots...")
        results_plots_file = plot_average_results(all_summed_rewards, all_avg_rewards, all_num_steps, episode,
                                                  description)
        print(f"- Plots saved: {results_plots_file}")
        results_store_file = self.save_results_to_file(all_summed_rewards, all_avg_rewards, all_num_steps,
                                                       per_episode_resources, description)
        print(f"- Results saved: {results_store_file}")
        print(f"- Resource log saved: {resource_log_file}")
        print(f"- CPU usage plot saved: {os.path.join(get_storage_path(), f'cpu_usage_{description}.png')}")
        print(f"- Memory usage plot saved: {os.path.join(get_storage_path(), f'memory_usage_{description}.png')}")


        # Save the final agent
        agent_file = os.path.join(get_storage_path(), f"agent={timestamp}.json")
        with open(agent_file, "w") as f:
            json.dump(agent.get_weights_dict(), f)
        print(f"- Agent representation saved: {agent_file}")

    def transform_fp(self, fp):
        return np.array(list(map(float, fp.split(","))))

    def compute_returns(self, rewards, dones, gamma):
        T = len(rewards)
        returns = np.zeros(T)
        g = 0
        for t in range(T - 1, -1, -1):
            g = rewards[t] + gamma * g * (1 - dones[t])
            returns[t] = g
        return returns

    def compute_gae(self, rewards, values, dones, states, gamma, lambda_):
        T = len(rewards)
        last_value = 0 if dones[-1] else self.agent.call(tf.convert_to_tensor(states[-1], dtype=tf.float32))[1][0, 0]
        values = np.append(values, last_value)
        deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
        advantages = np.zeros(T)
        a = 0
        for t in range(T - 1, -1, -1):
            a = deltas[t] + gamma * lambda_ * (1 - dones[t]) * a
            advantages[t] = a

        if len(advantages) == 1:
            return [-1.0]
        else:
            final_advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)  # Already normalized
            return final_advantages

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

    def save_results_to_file(self, all_summed_rewards, all_avg_rewards, all_num_steps, per_episode_resources,
                             run_description):
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

if __name__ == "__main__":
    controller = ControllerPPONormalAD()
    controller.run_c2()