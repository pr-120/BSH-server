import os

from environment.settings import TRAINING_CSV_FOLDER_PATH
from environment.state_handling import get_num_configs


metrics_dir = os.path.join(TRAINING_CSV_FOLDER_PATH, "metrics")
for config in range(get_num_configs()):
    with open(os.path.join(metrics_dir, "metrics-c{}.txt".format(config)), "r") as file:
        lines = file.readlines()[1:]  # drop headers
        sum = 0
        for line in lines:
            sum += float(line.split(",")[-2])
        avg = sum / len(lines)
        print("config", config, "avg %.3f" % avg, "for", sum, "out of", len(lines), "lines")
