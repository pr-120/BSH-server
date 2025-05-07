import os
import time

import numpy as np
import pandas as pd

from environment.anomaly_detection.anomaly_detection import detect_anomaly
from environment.anomaly_detection.constructor import get_preprocessor, reset_AD
from environment.reward.abstract_reward import AbstractReward
from environment.settings import TRAINING_CSV_FOLDER_PATH, ALL_CSV_HEADERS
from environment.state_handling import initialize_storage, cleanup_storage, set_prototype


def collect_fingerprint(d, f):
    p = os.path.join(d, f)
    with open(p, "r") as file:
        fp = file.readline().replace("[", "").replace("]", "").replace(" ", "")
    # print("Collected FP.")
    return fp


def transform_fp(fp):
    split_to_floats = list(map(lambda feat: float(feat), fp.split(",")))
    return np.asarray(split_to_floats)


def get_fp_features():
    df_normal = pd.read_csv(os.path.join(TRAINING_CSV_FOLDER_PATH, "normal-behavior.csv"))
    preprocessor = get_preprocessor()
    ready_dataset = preprocessor.preprocess_dataset(df_normal)
    return ready_dataset.columns


def preprocess_fp(fp, fp_features):
    headers = ALL_CSV_HEADERS.split(",")
    indexes = []
    for header in fp_features:
        indexes.append(headers.index(header))
    return fp[indexes]


initialize_storage()
print("========== PROTOTYPE 1 WITH SIMPLE PREPROCESSOR ==========")
set_prototype("1")
AbstractReward.prepare_reward_computation()
reset_AD()
print("\n========== PROTOTYPE 10 WITH ADVANCED PREPROCESSOR (IF) ==========")
set_prototype("10")
AbstractReward.prepare_reward_computation()
reset_AD()
print("\n========== PROTOTYPE 9 WITH ADVANCED PREPROCESSOR (AE) ==========")
set_prototype("9")
AbstractReward.prepare_reward_computation()

print("\n========== EVALUATE COLLECTED FP ==========")
p = os.path.join(os.path.abspath(os.path.curdir), "fingerprints")
files = os.listdir(p)
fp_features = get_fp_features()
ctr = 0
timer = time.time()
for f in files:
    fp = collect_fingerprint(p, f)
    t = transform_fp(fp)
    det = detect_anomaly(t)
    if det == 0:
        # print(f, det)
        ctr += 1
print(ctr, "/", len(files), "-- %.5f" % (ctr / (len(files) + 1e-10)))
print("Eval took %.3f" % (time.time() - timer), "seconds")
cleanup_storage()
