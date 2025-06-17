import tensorflow as tf
tf.keras.utils.set_random_seed(42)
import json
import os
import signal
from contextlib import redirect_stdout
from datetime import datetime
from multiprocessing import Process
from time import sleep, time
from v24.agent.agent import AgentPPONormalAD
from v24.environment.controller import ControllerPPONormalAD
from environment.reward.abstract_reward import AbstractReward
from environment.state_handling import initialize_storage, cleanup_storage, set_prototype, get_num_configs, \
    get_storage_path, set_simulation, get_instance_number, setup_child_instance, is_api_running, set_api_running
from environment.evaluation.evaluation_ppo import evaluate_agent  # Import for early stopping

"""
Want to change evaluated prototype?
1) adjust the known best action (setup) if necessary (depending on AD and reward system)
2) adjust the filename of the logfile (setup) to match the prototype settings (in controller/agent/model)
3) set prototype to evaluated prototype number in setup below (start of try-block)
4) set the parameters for the early stopping mechanism
"""


# Early stopping parameters
PATIENCE = 5  # Number of evaluation checks without improvement
EVAL_FREQ = 1000  # Evaluate every 1000 episodes

def create_app():
    from flask import Flask
    app = Flask(__name__)
    return app

def start_api(instance_number):
    setup_child_instance(instance_number)
    app = create_app()
    print("==============================\nStart API\n==============================")
    set_api_running()
    app.run(host="0.0.0.0", port=5000)

def kill_process(proc):
    print("kill Process", proc)
    proc.terminate()
    timeout = 10
    start = time()
    while proc.is_alive() and time() - start < timeout:
        sleep(1)
    if proc.is_alive():
        proc.kill()
        sleep(2)
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL)
            print("...die already", proc)
        else:
            print(proc, "now dead")
    else:
        print(proc, "now dead")

def find_agent_file(timestamp):
    """Find the saved agent representation file, preferring the best agent."""
    storage_path = get_storage_path()
    best_agent_file = os.path.join(storage_path, f"agent={timestamp}-best.json")
    final_agent_file = os.path.join(storage_path, f"agent={timestamp}.json")
    if os.path.exists(best_agent_file):
        return best_agent_file
    elif os.path.exists(final_agent_file):
        return final_agent_file
    else:
        raise FileNotFoundError(f"No agent file found for timestamp {timestamp}")

def print_accuracy_table(accuracies_overall, accuracies_configs, logs):
    """Print and log accuracy tables."""
    num_configs = get_num_configs()
    print("----- Per Config -----")
    logs.append("----- Per Config -----")

    line = []
    key_keys = sorted(list(filter(lambda k: not isinstance(k, str), list(accuracies_configs["normal"].keys()))))
    for key in range(num_configs):
        value = accuracies_configs["normal"][key] if key in key_keys else 0
        line.append("c{} {}% ({}/{})\t".format(key, "%05.2f" % (value / accuracies_configs["normal"]["total"] * 100),
                                               value, accuracies_configs["normal"]["total"]))
    print("Normal:\t", *line, sep="\t")
    logs.append("\t".join(["Normal:\t", *line]).strip())

    config_keys = sorted(list(filter(lambda k: not isinstance(k, str), list(accuracies_configs.keys()))))
    for config in config_keys:
        line = []
        key_keys = sorted(list(filter(lambda k: not isinstance(k, str), list(accuracies_configs[config].keys()))))
        for key in range(num_configs):
            value = accuracies_configs[config][key] if key in key_keys else 0
            line.append("c{} {}% ({}/{})\t".format(key, "%05.2f" % (value / accuracies_configs[config]["total"] * 100),
                                                   value, accuracies_configs[config]["total"]))
        print("Config {}:".format(config), *line, sep="\t")
        logs.append("\t".join(["Config {}:".format(config), *line]).strip())

    print("\n----- Overall -----")
    logs.append("\n----- Overall -----")

    overall_keys = sorted(list(filter(lambda k: not isinstance(k, str), list(accuracies_overall.keys()))))
    for config in range(num_configs):
        value = accuracies_overall[config] if config in overall_keys else 0
        print("Config {}:\t{}% ({}/{})".format(config, "%05.2f" % (value / accuracies_overall["total"] * 100), value,
                                               accuracies_overall["total"]))
        logs.append("Config {}:\t{}% ({}/{})".format(config, "%05.2f" % (value / accuracies_overall["total"] * 100),
                                                     value, accuracies_overall["total"]))

    return logs

if __name__ == "__main__":
    # ==============================
    # SETUP
    # ==============================
    total_start = time()
    prototype_description = "p25-100e=lr0.0001-clip0.1-g0.99-l0.95-EInit0.5-EDec0.995-Epochs5-batch32"
    KNOWN_BEST_ACTION = 3

    log_file = os.path.join(os.path.curdir, "storage",
                            f"accuracy-report={datetime.now().strftime('%Y-%m-%d--%H-%M-%S')}={prototype_description}.txt")
    logs = []

    print("========== PREPARE ENVIRONMENT ==========\nAD evaluation is written to log file directly")
    initialize_storage()
    procs = []
    try:
        prototype_num = "24"        #Set this to your Prototype number (24 or 25)
        set_prototype(prototype_num)
        simulated = True
        set_simulation(simulated)

        if prototype_num == "25":   #leave this as it is
            """AgentClass = AgentPPOIdealAD
            ControllerClass = ControllerPPOIdealAD"""
        else:
            AgentClass = AgentPPONormalAD
            ControllerClass = ControllerPPONormalAD

        with open(log_file, "w+") as f:
            with redirect_stdout(f):
                print("========== PREPARE ENVIRONMENT ==========")
                AbstractReward.prepare_reward_computation()

        # ==============================
        # EVAL UNTRAINED AGENT
        # ==============================
        print("\n========== MEASURE ACCURACY (INITIAL) ==========")
        logs.append("\n========== MEASURE ACCURACY (INITIAL) ==========")

        agent = AgentClass()
        #accuracies_initial_overall, accuracies_initial_configs = evaluate_agent(agent)
        print(f"Evaluating agent {agent} with settings {prototype_description}.\n")
        logs.append(f"Evaluating agent {agent} with settings {prototype_description}.\n")
        logs.append("Agent representation")
        logs.append("> prototype: PPO")

        weights_dict = agent.get_weights_dict()
        logs.append(f"> weights_input_hidden1: {weights_dict['weights_input_hidden1']}")
        logs.append(f"> weights_hidden1_hidden2: {weights_dict['weights_hidden1_hidden2']}")
        logs.append(f"> weights_hidden2_policy: {weights_dict['weights_hidden2_policy']}")
        logs.append(f"> weights_hidden2_value: {weights_dict['weights_hidden2_value']}")

        logs.append(f"> learn_rate: {agent.learn_rate}, clip_epsilon: {agent.clip_epsilon}, "
                    f"gamma: {agent.gamma}, lambda_: {agent.lambda_}, value_coef: {agent.value_coef}, "
                    f"entropy_coef: {agent.entropy_coef}, epochs: {agent.epochs}, batch_size: {agent.batch_size}")

        start = time()
        #accuracies_initial_overall, accuracies_initial_configs = evaluate_agent(agent)
        duration = time() - start
        print(f"\nEvaluation took %.3fs, roughly %.1fmin." % (duration, duration / 60))
        logs.append(f"\nEvaluation took %.3fs, roughly %.1fmin." % (duration, duration / 60))

        # ==============================
        # TRAINING AGENT
        # ==============================
        if not simulated:
            proc_api = Process(target=start_api, args=(get_instance_number(),))
            procs.append(proc_api)
            proc_api.start()
            print("\nWaiting for API...")
            while not is_api_running():
                sleep(1)

        print("\n========== TRAIN AGENT ==========")
        logs.append("\n========== TRAIN AGENT ==========")

        controller = ControllerClass()
        training_start = time()
        timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        controller.run_c2(agent, timestamp=timestamp, patience=PATIENCE, eval_freq=EVAL_FREQ)
        training_duration = time() - training_start
        logs.append(f"Agent and plots timestamp: {timestamp}")
        print(f"Training took %.3fs, roughly %.1fmin." % (training_duration, training_duration / 60))
        logs.append(f"Training took %.3fs, roughly %.1fmin." % (training_duration, training_duration / 60))

        # ==============================
        # EVAL TRAINED AGENT
        # ==============================
        print("\n========== MEASURE ACCURACY (TRAINED) ==========")
        logs.append("\n========== MEASURE ACCURACY (TRAINED) ==========")

        with open(find_agent_file(timestamp), "r") as agent_file:
            repr_dict = json.load(agent_file)
        agent = AgentClass(representation=repr_dict)

        logs.append("Trained agent representation")
        logs.append("> prototype: PPO")

        weights_dict = agent.get_weights_dict()
        logs.append(f"> weights_input_hidden1: {weights_dict['weights_input_hidden1']}")
        logs.append(f"> weights_hidden1_hidden2: {weights_dict['weights_hidden1_hidden2']}")
        logs.append(f"> weights_hidden2_policy: {weights_dict['weights_hidden2_policy']}")
        logs.append(f"> weights_hidden2_value: {weights_dict['weights_hidden2_value']}")

        logs.append(f"> learn_rate: {agent.learn_rate}, clip_epsilon: {agent.clip_epsilon}, "
                    f"gamma: {agent.gamma}, lambda_: {agent.lambda_}, value_coef: {agent.value_coef}, "
                    f"entropy_coef: {agent.entropy_coef}, epochs: {agent.epochs}, batch_size: {agent.batch_size}")

        start = time()
        accuracies_trained_overall, accuracies_trained_configs = evaluate_agent(agent)
        duration = time() - start
        print(f"\nEvaluation took %.3fs, roughly %.1fmin." % (duration, duration / 60))
        logs.append(f"\nEvaluation took %.3fs, roughly %.1fmin." % (duration, duration / 60))

        # Compute accuracy tables
        print("\n========== ACCURACY TABLE (INITIAL) ==========")
        logs.append("\n========== ACCURACY TABLE (INITIAL) ==========")
        #logs = print_accuracy_table(accuracies_initial_overall, accuracies_initial_configs, logs)

        print("\n========== ACCURACY TABLE (TRAINED) ==========")
        logs.append("\n========== ACCURACY TABLE (TRAINED) ==========")
        logs = print_accuracy_table(accuracies_trained_overall, accuracies_trained_configs, logs)

        # Show results
        print("\n========== RESULTS ==========")
        logs.append("\n========== RESULTS ==========")

        """val_initial = accuracies_initial_overall.get(KNOWN_BEST_ACTION, 0)
        known_best_initial = "{}% ({}/{})".format("%05.2f" % (val_initial / accuracies_initial_overall["total"] * 100),
                                                  val_initial, accuracies_initial_overall["total"])
        val_trained = accuracies_trained_overall.get(KNOWN_BEST_ACTION, 0)
        known_best_trained = "{}% ({}/{})".format("%05.2f" % (val_trained / accuracies_trained_overall["total"] * 100),
                                                  val_trained, accuracies_trained_overall["total"])
        print(f"For known best action {KNOWN_BEST_ACTION}: from {known_best_initial} to {known_best_trained}.")
        logs.append(f"For known best action {KNOWN_BEST_ACTION}: from {known_best_initial} to {known_best_trained}.")"""

        total_duration = time() - total_start
        print(f"Accuracy computation took %.3fs in total, roughly %.1fmin." % (total_duration, total_duration / 60))
        logs.append(f"Accuracy computation took %.3fs in total, roughly %.1fmin." % (total_duration, total_duration / 60))

        with open(log_file, "a") as file:
            file.writelines([l + "\n" for l in logs])

    finally:
        for proc in procs:
            kill_process(proc)
        print("- Parallel processes killed.")
        cleanup_storage()
        print("- Storage cleaned up.\n==============================")