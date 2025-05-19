import os
import random

from environment.state_handling import get_num_configs
from environment.settings import TRAINING_CSV_FOLDER_PATH, EVALUATION_CSV_FOLDER_PATH, FINGERPRINT_FOLDER_PATH


evaluation_dir = os.path.join(FINGERPRINT_FOLDER_PATH, "evaluation")  # path to target folder for test sets
training_dir = os.path.join(FINGERPRINT_FOLDER_PATH, "training")  # path to target folder for training sets


def move_files(origin_dir, evaluation_target):
    remaining_files = os.listdir(origin_dir)
    size = len(remaining_files)

    os.makedirs(evaluation_target, exist_ok=True)
    for i in range(int(0.2 * size)):
        file = random.choice(remaining_files)
        os.rename(os.path.join(origin_dir, file), os.path.join(evaluation_target, file))
        remaining_files.remove(file)


for config in range(get_num_configs()):
    print("Moving config", config)
    origin_dir = os.path.join(training_dir, "infected-c{}".format(config))
    evaluation_target = os.path.join(evaluation_dir, "infected-c{}".format(config))
    training_target = os.path.join(training_dir, "infected-c{}".format(config))
    move_files(origin_dir, evaluation_target)

print("Moving normal")
origin_dir = os.path.join(training_dir, "normal")
evaluation_target = os.path.join(evaluation_dir, "normal")
training_target = os.path.join(training_dir, "normal")
move_files(origin_dir, evaluation_target)

print("Done")
