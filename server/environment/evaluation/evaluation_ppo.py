import numpy as np
import os
from tqdm import tqdm
from environment.settings import EVALUATION_CSV_FOLDER_PATH, FINGERPRINT_FOLDER_PATH
from environment.state_handling import get_num_configs

def transform_fp(fp):
    """Convert fingerprint string to NumPy array, matching controller.py."""
    return np.array(list(map(float, fp.split(","))))  # Shape: (F,)

def evaluate_agent(agent):
    """Evaluate the agent's accuracy on normal and infected fingerprints."""
    accuracies_overall = {"total": 0}
    accuracies_configs = {}
    num_configs = get_num_configs()

    for config in range(num_configs):
        accuracies_overall[config] = 0

    # Evaluate normal fingerprints
    print("Normal")
    normal_fp_dir = os.path.join(FINGERPRINT_FOLDER_PATH, "evaluation", "normal")
    fp_files = os.listdir(normal_fp_dir)
    accuracies_configs["normal"] = {"total": 0}
    for fp_file in tqdm(fp_files):
        with open(os.path.join(normal_fp_dir, fp_file)) as file:
            fp = file.readline()[1:-1].replace(" ", "")
        state = transform_fp(fp)
        selected_action = agent.evaluate_action(state)
        accuracies_overall[selected_action] = accuracies_overall.get(selected_action, 0) + 1
        accuracies_overall["total"] += 1
        accuracies_configs["normal"][selected_action] = accuracies_configs["normal"].get(selected_action, 0) + 1
        accuracies_configs["normal"]["total"] += 1

    # Evaluate infected fingerprints
    for config in range(num_configs):
        print(f"\nConfig {config}")
        config_fp_dir = os.path.join(FINGERPRINT_FOLDER_PATH, "evaluation", f"infected-c{config}")
        fp_files = os.listdir(config_fp_dir)
        accuracies_configs[config] = {"total": 0}
        for fp_file in tqdm(fp_files):
            with open(os.path.join(config_fp_dir, fp_file)) as file:
                fp = file.readline()[1:-1].replace(" ", "")
            state = transform_fp(fp)
            selected_action = agent.evaluate_action(state)
            accuracies_overall[selected_action] = accuracies_overall.get(selected_action, 0) + 1
            accuracies_overall["total"] += 1
            accuracies_configs[config][selected_action] = accuracies_configs[config].get(selected_action, 0) + 1
            accuracies_configs[config]["total"] += 1

    return accuracies_overall, accuracies_configs