import os
from datetime import datetime
from threading import Lock

from environment.state_handling import set_fp_ready, get_fp_file_path, get_rate_file_path


def write_metrics_to_file(fp, storage_path, is_multi):
    # print("UTILS: writing rate/fp", rate, fp, is_multi)
    __write_fingerprint_to_file(fp, storage_path, is_multi)
    # print("UTILS: rate/fp written")
    set_fp_ready(True)


def __write_rate_to_file(rate, storage_path):
    os.makedirs(storage_path, exist_ok=True)
    rate_path = get_rate_file_path()

    lock = Lock()
    with lock:
        with open(rate_path, "w") as file:
            file.write(str(rate))


def __write_fingerprint_to_file(fp, storage_path, is_multi):
    os.makedirs(storage_path, exist_ok=True)
    if is_multi:
        file_name = "fp-{time}.txt".format(time=datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        fp_path = os.path.join(storage_path, file_name)
    else:
        fp_path = get_fp_file_path()

    lock = Lock()
    with lock:
        try:
            with open(fp_path, "x" if is_multi else "w") as file:
                file.write(fp)

        # with multiple devices returning fingerprints at the same time it can happen
        # that there are multiple fps at the same time, resulting in same names
        except FileExistsError:

            while True:
                path_addition = 1

                try:
                    # retry with adjusted filename
                    with open(fp_path + "(" + str(path_addition) + ")", "x" if is_multi else "w") as file:
                        file.write(fp)
                        break   # break when file was successfully created
                except FileExistsError:
                    path_addition += 1