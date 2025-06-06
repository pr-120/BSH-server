import json
import os
import random

from environment.settings import FINGERPRINT_FOLDER_PATH
from environment.state_handling import get_storage_path, is_multi_fp_collection, set_rw_done
from utilities.metrics import write_metrics_to_file

# ==============================
# SIMULATE CLIENT BEHAVIOR
# ==============================

UNLIMITED_CONFIGURATIONS = [1, 2]
AVERAGE_RATES = {  # calculated by auxiliary script ´find_avg_rate.py´
    1: 565565.651186441,
    2: 632834.8006,
}


def simulate_sending_fp(config_num):
    config_fp_dir = os.path.join(FINGERPRINT_FOLDER_PATH, "training", "infected-c{}".format(config_num))
    fp_files = os.listdir(config_fp_dir)
    with open(os.path.join(config_fp_dir, random.choice(fp_files))) as fp_file:
        # print("SIM: fp", fp_file.name)
        fp = fp_file.read()

    write_metrics_to_file(fp, get_storage_path(), is_multi_fp_collection())


def simulate_sending_rw_done():
    set_rw_done()
