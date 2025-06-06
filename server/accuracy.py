import json
import os
import signal
from contextlib import redirect_stdout
from datetime import datetime
from multiprocessing import Process
from time import sleep, time

import numpy as np
from tqdm import tqdm

from agent.agent_representation import AgentRepresentation
from agent.constructor import get_agent, build_agent_from_repr
from api import create_app
from environment.constructor import get_controller
from environment.reward.abstract_reward import AbstractReward
from environment.settings import FINGERPRINT_FOLDER_PATH
from environment.state_handling import initialize_storage, cleanup_storage, set_prototype, get_num_configs, \
    get_storage_path, set_simulation, get_instance_number, setup_child_instance, is_api_running, set_api_running

"""
Want to change evaluated prototype?
1) adjust initial EPSILON (setup) to the one used in evaluated prototype (usually in controller)
2) adjust the known best action (setup) if necessary (depending on AD and reward system)
3) adjust the filename of the logfile (setup) to match the prototype settings (in controller/agent/model)
4) set prototype to evaluated prototype number in setup below (start of try-block)
5) adjust simulation settings if required (start of try-block)
"""


def start_api(instance_number):
    setup_child_instance(instance_number)
    app = create_app()
    print("==============================\nStart API\n==============================")
    set_api_running()
    app.run(host="0.0.0.0", port=5000)


def kill_process(proc):
    print("kill Process", proc)
    proc.terminate()
    print("killed Process", proc)
    timeout = 10
    start = time()
    while proc.is_alive() and time() - start < timeout:
        sleep(1)
    if proc.is_alive():
        proc.kill()
        print("...we had to put it down", proc)
        sleep(2)
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL)
            print("...die already", proc)
        else:
            print(proc, "now dead")
    else:
        print(proc, "now dead")


def evaluate_agent(agent, weights1, weights2, bias_weights1, bias_weights2, EPSILON):
    accuracies_overall = {"total": 0}
    accuracies_configs = {}
    num_configs = get_num_configs()

    for config in range(num_configs):  # ensure all keys exist for accessing through KNOWN_BEST_ACTION
        accuracies_overall[config] = 0

    # eval normal
    print("Normal")
    normal_fp_dir = os.path.join(FINGERPRINT_FOLDER_PATH, "evaluation", "normal")
    fp_files = os.listdir(normal_fp_dir)
    accuracies_configs["normal"] = {"total": 0}
    for fp_file in tqdm(fp_files):
        # collect selected initial fingerprint
        with open(os.path.join(normal_fp_dir, fp_file)) as file:
            fp = file.readline()[1:-1].replace(" ", "")
        state = transform_fp(fp)

        # predict next action
        curr_hidden, curr_q_values, selected_action = agent.predict(weights1, weights2, bias_weights1,
                                                                    bias_weights2, EPSILON, state)

        if selected_action not in accuracies_overall.keys():
            accuracies_overall[selected_action] = 1
        else:
            accuracies_overall[selected_action] += 1
        accuracies_overall["total"] += 1

        if selected_action not in accuracies_configs["normal"].keys():
            accuracies_configs["normal"][selected_action] = 1
        else:
            accuracies_configs["normal"][selected_action] += 1
        accuracies_configs["normal"]["total"] += 1

    # eval infected
    for config in range(num_configs):
        print("\nConfig", config)
        config_fp_dir = os.path.join(FINGERPRINT_FOLDER_PATH, "evaluation", "infected-c{}".format(config))
        fp_files = os.listdir(config_fp_dir)

        accuracies_configs[config] = {"total": 0}
        for fp_file in tqdm(fp_files):
            # collect selected initial fingerprint
            with open(os.path.join(config_fp_dir, fp_file)) as file:
                fp = file.readline()[1:-1].replace(" ", "")
            state = transform_fp(fp)

            # predict next action
            curr_hidden, curr_q_values, selected_action = agent.predict(weights1, weights2, bias_weights1,
                                                                        bias_weights2, EPSILON, state)

            if selected_action not in accuracies_overall.keys():
                accuracies_overall[selected_action] = 1
            else:
                accuracies_overall[selected_action] += 1
            accuracies_overall["total"] += 1

            if selected_action not in accuracies_configs[config].keys():
                accuracies_configs[config][selected_action] = 1
            else:
                accuracies_configs[config][selected_action] += 1
            accuracies_configs[config]["total"] += 1
    return accuracies_overall, accuracies_configs


def find_agent_file(timestamp):
    storage_path = get_storage_path()
    files = os.listdir(storage_path)
    filtered = list(filter(lambda f: f.startswith("agent={}".format(timestamp)), files))
    return os.path.join(storage_path, filtered.pop())


def transform_fp(fp):
    split_to_floats = list(map(lambda feat: float(feat), fp.split(",")))
    return np.asarray(split_to_floats).reshape(-1, 1)  # shape (F, 1)


def print_accuracy_table(accuracies_overall, accuracies_configs, logs):
    num_configs = get_num_configs()
    print("----- Per Config -----")
    logs.append("----- Per Config -----")

    # from normal states
    line = []
    key_keys = sorted(list(filter(lambda k: not isinstance(k, str), list(accuracies_configs["normal"].keys()))))
    for key in range(num_configs):
        value = accuracies_configs["normal"][key] if key in key_keys else 0
        line.append("c{} {}% ({}/{})\t".format(key, "%05.2f" % (value / accuracies_configs["normal"]["total"] * 100),
                                               value, accuracies_configs["normal"]["total"]))
    print("Normal:\t", *line, sep="\t")
    logs.append("\t".join(["Normal:\t", *line]).strip())

    # from infected states
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
    prototype_description = "p20-100e=e0.5d0.1a0.0050y0.10=Log-SiLU=h40=he"

    EPSILON = 0.5
    KNOWN_BEST_ACTION = 3

    log_file = os.path.join(os.path.curdir, "storage",
                            "accuracy-report={}={}.txt".format(datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),
                                                               prototype_description))
    logs = []

    print("========== PREPARE ENVIRONMENT ==========\nAD evaluation is written to log file directly")

    initialize_storage()
    procs = []
    try:
        set_prototype("20")
        simulated = True
        set_simulation(simulated)
        np.random.seed(42)

        # ==============================
        # WRITE AD EVALUATION TO LOG FILE
        # ==============================

        with open(log_file, "w+") as f:
            with redirect_stdout(f):
                print("========== PREPARE ENVIRONMENT ==========")
                AbstractReward.prepare_reward_computation()

        # ==============================
        # EVAL UNTRAINED AGENT
        # ==============================

        print("\n========== MEASURE ACCURACY (INITIAL) ==========")
        logs.append("\n========== MEASURE ACCURACY (INITIAL) ==========")

        agent = get_agent()
        print("Evaluating agent {} with settings {}.\n".format(agent, prototype_description))
        logs.append("Evaluating agent {} with settings {}.\n".format(agent, prototype_description))
        weights1, weights2, bias_weights1, bias_weights2 = agent.initialize_network()
        logs.append("Agent representation")
        logs.append("> prototype: {}".format(agent))
        logs.append("> weights1: {}".format(weights1.tolist()))
        logs.append("> weights2: {}".format(weights2.tolist()))
        logs.append("> bias_weights1: {}".format(bias_weights1.tolist()))
        logs.append("> bias_weights2: {}".format(bias_weights2.tolist()))
        logs.append("> epsilon: {}, learn_rate: {}, num_input: {}, num_hidden: {}, num_output: {}".format(
            EPSILON, agent.learn_rate, agent.num_input, agent.num_hidden, agent.num_output))

        start = time()
        accuracies_initial_overall, accuracies_initial_configs = evaluate_agent(agent, weights1, weights2,
                                                                                bias_weights1,
                                                                                bias_weights2, EPSILON)
        duration = time() - start
        print("\nEvaluation took {}s, roughly {}min.".format("%.3f" % duration, "%.1f" % (duration / 60)))
        logs.append("\nEvaluation took {}s, roughly {}min.".format("%.3f" % duration, "%.1f" % (duration / 60)))

        # ==============================
        # TRAINING AGENT
        # ==============================

        if not simulated:
            # Start API listener
            proc_api = Process(target=start_api, args=(get_instance_number(),))
            procs.append(proc_api)
            proc_api.start()
            print("\nWaiting for API...")
            while not is_api_running():
                sleep(1)

        print("\n========== TRAIN AGENT ==========")
        logs.append("\n========== TRAIN AGENT ==========")

        controller = get_controller()
        training_start = time()
        timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        final_q_values, _ = controller.loop_episodes(agent)
        training_duration = time() - training_start
        logs.append("Agent and plots timestamp: {}".format(timestamp))
        print("Training took {}s, roughly {}min.".format("%.3f" % training_duration, "%.1f" % (training_duration / 60)))
        logs.append("Training took {}s, roughly {}min.".format("%.3f" % training_duration,
                                                               "%.1f" % (training_duration / 60)))

        # ==============================
        # EVAL TRAINED AGENT
        # ==============================

        print("\n========== MEASURE ACCURACY (TRAINED) ==========")
        logs.append("\n========== MEASURE ACCURACY (TRAINED) ==========")

        with open(find_agent_file(timestamp), "r") as agent_file:
            repr_dict = json.load(agent_file)
        representation = (
            repr_dict["weights1"], repr_dict["weights2"], repr_dict["bias_weights1"], repr_dict["bias_weights2"],
            repr_dict["epsilon"], repr_dict["learn_rate"], repr_dict["num_input"], repr_dict["num_hidden"],
            repr_dict["num_output"]
        )
        agent = build_agent_from_repr(AgentRepresentation(*representation))
        final_epsilon = repr_dict["epsilon"]
        weights1, weights2, bias_weights1, bias_weights2 = agent.initialize_network()
        logs.append("Agent representation")
        logs.append("> prototype: {}".format(agent))
        logs.append("> weights1: {}".format(weights1.tolist()))
        logs.append("> weights2: {}".format(weights2.tolist()))
        logs.append("> bias_weights1: {}".format(bias_weights1.tolist()))
        logs.append("> bias_weights2: {}".format(bias_weights2.tolist()))
        logs.append("> epsilon: {}, learn_rate: {}, num_input: {}, num_hidden: {}, num_output: {}".format(
            final_epsilon, agent.learn_rate, agent.num_input, agent.num_hidden, agent.num_output))

        start = time()
        accuracies_trained_overall, accuracies_trained_configs = evaluate_agent(agent, weights1, weights2,
                                                                                bias_weights1,
                                                                                bias_weights2, final_epsilon)
        duration = time() - start
        print("\nEvaluation took {}s, roughly {}min.".format("%.3f" % duration, "%.1f" % (duration / 60)))
        logs.append("\nEvaluation took {}s, roughly {}min.".format("%.3f" % duration, "%.1f" % (duration / 60)))

        # ==============================
        # COMPUTE ACCURACY TABLES
        # ==============================

        print("\n========== ACCURACY TABLE (INITIAL) ==========")
        logs.append("\n========== ACCURACY TABLE (INITIAL) ==========")
        logs = print_accuracy_table(accuracies_initial_overall, accuracies_initial_configs, logs)

        print("\n========== ACCURACY TABLE (TRAINED) ==========")
        logs.append("\n========== ACCURACY TABLE (TRAINED) ==========")
        logs = print_accuracy_table(accuracies_trained_overall, accuracies_trained_configs, logs)

        # ==============================
        # SHOW RESULTS
        # ==============================

        print("\n========== RESULTS ==========")
        logs.append("\n========== RESULTS ==========")

        # highlight change through training
        val_initial = accuracies_initial_overall[KNOWN_BEST_ACTION]
        known_best_initial = "{}% ({}/{})".format("%05.2f" % (val_initial / accuracies_initial_overall["total"] * 100),
                                                  val_initial, accuracies_initial_overall["total"])
        val_trained = accuracies_trained_overall[KNOWN_BEST_ACTION]
        known_best_trained = "{}% ({}/{})".format("%05.2f" % (val_trained / accuracies_initial_overall["total"] * 100),
                                                  val_trained, accuracies_initial_overall["total"])
        print("For known best action {}: from {} to {}.".format(KNOWN_BEST_ACTION, known_best_initial,
                                                                known_best_trained))
        logs.append("For known best action {}: from {} to {}.".format(KNOWN_BEST_ACTION, known_best_initial,
                                                                      known_best_trained))

        # report final Q-values
        print("Final Q-Values:\n{}".format(final_q_values))
        logs.append("Final Q-Values:\n{}".format(final_q_values))

        # show time required for entire accuracy computation
        total_duration = time() - total_start
        print("Accuracy computation took {}s in total, roughly {}min.".format("%.3f" % total_duration,
                                                                              "%.1f" % (total_duration / 60)))
        logs.append("Accuracy computation took {}s in total, roughly {}min.".format("%.3f" % total_duration,
                                                                                    "%.1f" % (total_duration / 60)))

        # ==============================
        # WRITE LOGS TO LOG FILE
        # ==============================

        with open(log_file, "a") as file:
            log_lines = list(map(lambda l: l + "\n", logs))
            file.writelines(log_lines)
    finally:
        for proc in procs:
            kill_process(proc)
        print("- Parallel processes killed.")
        cleanup_storage()
        print("- Storage cleaned up.\n==============================")
