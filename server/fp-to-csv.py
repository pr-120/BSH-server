from copy import deepcopy
import os

from environment.settings import EVALUATION_CSV_FOLDER_PATH, FINGERPRINT_FOLDER_PATH, TRAINING_CSV_FOLDER_PATH, \
    ALL_CSV_HEADERS
from environment.state_handling import get_num_configs

# CSV_FOLDER_PATH = EVALUATION_CSV_FOLDER_PATH
CSV_FOLDER_PATH = TRAINING_CSV_FOLDER_PATH

# FP directories
normal_fp_dir = os.path.join(FINGERPRINT_FOLDER_PATH, "training/normal")
fp_dirs = [normal_fp_dir]
for conf_nr in range(get_num_configs()):
    infected_conf_fp_dir = os.path.join(FINGERPRINT_FOLDER_PATH, "infected-c{}".format(conf_nr))
    fp_dirs.append(infected_conf_fp_dir)

# headers based on FP script fingerprinter.sh
CSV_HEADERS = ALL_CSV_HEADERS


def find_duplicate_headers():
    headers = CSV_HEADERS.split(",")
    print("- original:", len(headers))
    unique = set(headers)
    print("- unique:", len(unique))
    diff = deepcopy(headers)
    for head in unique:
        diff.remove(head)
    print("- duplicates:", len(diff), diff)


def prepare_csv_file(behavior):
    csv_file_name = "{}-behavior.csv".format(behavior)
    csv_file_path = os.path.join(CSV_FOLDER_PATH, csv_file_name)

    if os.path.exists(csv_file_path):
        print("Removing existing CSV file", csv_file_name)
        os.remove(csv_file_path)

    print("Creating new CSV file", csv_file_name)
    with open(csv_file_path, "x"):
        pass

    return csv_file_path


def write_contents(contents, file_path):
    with open(file_path, "w") as file:
        file.write(CSV_HEADERS + "\n")
        for line in contents:
            file.write(line + "\n")


def verify_contents(file_path):
    header_length = len(CSV_HEADERS.split(","))
    with open(file_path, "r") as file:
        for line in file:
            line_length = len(line.split(","))
            assert line_length == header_length, \
                "Line length {} did not match header length {} in line {}".format(line_length, header_length, line)
        print("Verification: all good.")


if __name__ == "__main__":
    print("Find duplicate headers.")
    find_duplicate_headers()

    print("Reading file contents.")
    for directory in fp_dirs:
        files = os.listdir(directory)
        all_lines = []
        for file in files:
            file_path = os.path.join(directory, file)
            with open(file_path, "r") as f:
                fp = f.readline().replace("[", "").replace("]", "").replace(" ", "")
                all_lines.append(fp)

        behavior = os.path.basename(directory)
        print("Preparing CSV file for", behavior, "behavior.")
        csv_file_path = prepare_csv_file(behavior)

        print("Writing contents to CSV.")
        write_contents(all_lines, csv_file_path)
        print("Verifying CSV contents.")
        verify_contents(csv_file_path)
        print("Done with", behavior, "behavior.")
