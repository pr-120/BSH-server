import os
import random

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from environment.anomaly_detection.constructor import get_preprocessor, get_classifier
from environment.settings import EVALUATION_CSV_FOLDER_PATH, TRAINING_CSV_FOLDER_PATH, ALL_CSV_HEADERS, \
    DUPLICATE_HEADERS
from environment.state_handling import get_num_configs

SCALER = None


def __init_scaler(train_set):
    global SCALER
    SCALER = StandardScaler().fit(train_set)  # Feature scaling


def __get_scaler():
    global SCALER
    assert SCALER is not None, "Must first initialize scaler and fit to training set!"
    return SCALER


def prepare_training_test_sets(train_set, test_set):
    # print("prep", train_set.shape, test_set.shape)

    ratio = len(train_set) / len(test_set)

    # automatically adjust to needed ratio
    while not (4 + 1e-5 >= ratio >= 4 - 1e-5):  # 0.8 to 0.2 has factor 4 +- some epsilon

        while ratio > 4 + 1e-5:
            train_set.drop(axis=0, index=(random.randint(0, len(train_set) - 1)), inplace=True)
            ratio = len(train_set) / len(test_set)

        while ratio < 4 - 1e-5:
            test_set.drop(axis=0, index=(random.randint(0, len(test_set) - 1)), inplace=True)
            ratio = len(train_set) / len(test_set)

    # Remove train data with Z-score higher than 3
    train_set = train_set[(np.abs(stats.zscore(train_set)) < 1000).all(axis=1)]

    return train_set, test_set


def scale_dataset(scaler, dataset):
    dataset = scaler.transform(dataset)
    return dataset


def map_results_to_category(results):
    # Goal: detect infected behavior; positive = infected, negative = normal
    true_positives = false_positives = true_negatives = false_negatives = 0
    for run in range(len(results)):
        data = results[run]
        elements = data[0]
        numbers = data[1]
        normal = data[2]
        # print("MAP", run, elements, numbers, infected)
        if len(elements) > 1:  # detected normal and infected samples
            if normal:
                true_negatives += numbers[0]
                false_positives += numbers[1]
            else:
                false_negatives += numbers[0]
                true_positives += numbers[1]
        else:  # detected only one type of samples
            if elements[0] == 0:  # detected normal
                if normal:
                    true_negatives += numbers[0]  # detected normal and is normal
                else:
                    false_negatives += numbers[0]  # detected normal but is infected
            else:  # detected infected
                if normal:
                    false_positives += numbers[0]  # detected infected but is normal
                else:
                    true_positives += numbers[0]  # detected infected and is infected
    return true_positives, true_negatives, false_positives, false_negatives


def calculate_f1_score(true_positives, true_negatives, false_positives, false_negatives):
    # print(true_positives, true_negatives, false_positives, false_negatives,
    #       true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    # print(precision, recall)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def evaluate_dataset(name, dataset):
    clf = get_classifier()
    pred = clf.predict(dataset)
    unique_elements, counts_elements = np.unique(pred, return_counts=True)
    if len(counts_elements) > 1:
        perc = (counts_elements[0] / (counts_elements[0] + counts_elements[1]) * 100)
    elif unique_elements[0] == 1:  # detected
        perc = 0
    else:  # hidden
        perc = 100
    print(name, unique_elements, counts_elements, "%.2f" % perc, sep="\t")
    return unique_elements, counts_elements


def train_anomaly_detection():
    # ==============================
    # LOAD, PROCESS, EVALUATE NORMAL DATA
    # ==============================

    # Load data
    # print("Loading CSV data.")
    csv_path_template = os.path.join(TRAINING_CSV_FOLDER_PATH, "{}-behavior.csv")
    df_normal_train = pd.read_csv(csv_path_template.format("normal"))
    df_normal_test = pd.read_csv(os.path.join(EVALUATION_CSV_FOLDER_PATH, "{}-behavior.csv").format("normal"))

    # randomly sample all rows to mimic shuffling
    df_normal_train = df_normal_train.sample(frac=1, random_state=42).reset_index(drop=True)
    # print("load", df_normal_train.shape, df_normal_test.shape)

    # Preprocess data for ML
    # print("Preprocessing datasets.")
    preprocessor = get_preprocessor()
    normal_train_data = preprocessor.preprocess_dataset(df_normal_train)
    normal_test_data = preprocessor.preprocess_dataset(df_normal_test)
    # print("proc", normal_train_data.shape, normal_test_data.shape)

    # print("Prepare normal behavior data training and test set.")
    train_set, test_set = prepare_training_test_sets(normal_train_data, normal_test_data)
    # print("sets", train_set.shape, test_set.shape)

    # Scale the datasets, turning them into ndarrays
    # print("Scaling dataset features to fit training set.")
    __init_scaler(train_set)
    scaler = __get_scaler()
    train_set = scale_dataset(scaler, train_set)
    test_set = scale_dataset(scaler, test_set)
    # print("scaled", train_set.shape, test_set.shape)

    # Instantiate ML Isolation Forest instance
    # print("Instantiate classifier.")
    clf = get_classifier()

    # Train model
    # print("Train classifier on training set.")
    clf.fit(train_set)

    # Evaluate model
    print("Evaluate test set and infected behavior datasets.")
    normal_results = evaluate_dataset("normal", test_set)

    # ==============================
    # REPEAT FOR INFECTED SAMPLES
    # ==============================

    all_results = [[*normal_results, True]]
    for conf_nr in range(get_num_configs()):
        df_inf = pd.read_csv(csv_path_template.format("infected-c{}".format(conf_nr)))
        inf_data = preprocessor.preprocess_dataset(df_inf)
        inf_data = scale_dataset(scaler, inf_data)
        inf_results = evaluate_dataset("inf-c{}".format(conf_nr), inf_data)
        all_results.append([*inf_results, False])
    f1 = calculate_f1_score(*map_results_to_category(all_results))
    print("Overall F1 score: %.5f" % f1)


def detect_anomaly(fingerprint):  # string
    # print("Detecting anomaly.")

    # Transforming FP string to pandas DataFrame
    fp_data = fingerprint.reshape(1, -1)

    headers = ALL_CSV_HEADERS.split(",")
    for header in DUPLICATE_HEADERS:
        found = headers.index(header)
        headers[found + 1] = headers[found + 1] + ".1"  # match the .1 for duplicates appended by read_csv()

    df_fp = pd.DataFrame(fp_data, columns=headers)

    # Sanitizing FP to match IsolationForest
    preprocessor = get_preprocessor()
    preprocessed = preprocessor.preprocess_dataset(df_fp)
    scaler = __get_scaler()
    scaled = scale_dataset(scaler, preprocessed)
    # print("Scaled FP to", scaled.shape)

    # Evaluate fingerprint
    clf = get_classifier()
    pred = clf.predict(scaled)
    assert type(pred) == np.ndarray and len(pred) == 1
    return pred[0]
