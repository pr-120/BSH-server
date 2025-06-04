import json
import os
from threading import Lock

from tinydb import TinyDB, Query
from tinydb.operations import set
from dotenv import load_dotenv

# loads environment
current_folder = os.path.dirname(os.path.abspath(__file__))
CONFIG_FOLDER = os.path.join(current_folder, "../../config")
load_dotenv(os.path.join(CONFIG_FOLDER, "folder_paths.config"))

STORAGE_FOLDER_NAME = "storage"
MULTI_FP_COLLECTION_FOLDER_NAME = "fingerprints"


# ==============================
# EXECUTION
# ==============================
def is_fp_ready():
    return __query_key("FP_READY")


def set_fp_ready(ready_state):
    __set_value("FP_READY", ready_state)


def is_rw_done():
    return __query_key("RW_DONE")


def set_rw_done(done=True):
    __set_value("RW_DONE", done)


def get_num_configs():
    num_configs = len(os.listdir(os.path.join(current_folder, "../bd-configs")))
    return num_configs


def collect_fingerprint():
    with open(get_fp_file_path(), "r") as file:
        fp = file.readline()[1:-1].replace(" ", "")
    # print("Collected FP.")
    return fp


def collect_rate() -> float:
    rate_file = os.path.join(CONFIG_FOLDER, "current_configuration.json")
    with open(rate_file, "r") as file:
        action = json.load(file)["current_configuration"]

    config = map_to_backdoor_configuration(action)
    buffer_size = float(config["buffer_size"])
    transfer_frequency = float(config["transfer_frequency"])
    rate = int(buffer_size) / transfer_frequency
    return rate


def set_agent_representation_path(path):
    __set_value("AGENT_REPR", path)


def get_agent_representation_path():
    return __query_key("AGENT_REPR")


# ==============================
# ORCHESTRATION
# ==============================
def is_multi_fp_collection():
    return __query_key("COLLECT_MULTIPLE_FP")


def set_multi_fp_collection(is_multi):
    __set_value("COLLECT_MULTIPLE_FP", is_multi)


def get_prototype():
    return __query_key("PROTOTYPE")


def set_prototype(proto):
    __set_value("PROTOTYPE", proto)


def is_simulation():
    return __query_key("SIMULATION")


def set_simulation(simulated):
    __set_value("SIMULATION", simulated)


def is_api_running():
    return __query_key("API")


def set_api_running():
    __set_value("API", True)


# ==============================
# STATE STORAGE
# ==============================
INSTANCE_NUMBER = None


def get_instance_number():
    return INSTANCE_NUMBER


def setup_child_instance(parent_instance):
    global INSTANCE_NUMBER
    INSTANCE_NUMBER = parent_instance


def initialize_storage():
    __determine_instance()
    __prepare_storage_file()
    db = __get_storage()
    db.drop_tables()  # reset database to start from scratch
    db.insert({"key": "COLLECT_MULTIPLE_FP", "value": False})
    db.insert({"key": "FP_READY", "value": False})
    db.insert({"key": "RW_DONE", "value": False})
    db.insert({"key": "PROTOTYPE", "value": 0})
    db.insert({"key": "SIMULATION", "value": False})
    db.insert({"key": "API", "value": False})
    db.insert({"key": "AGENT_REPR", "value": ""})


def cleanup_storage():
    storage_file, _ = __get_storage_file_path()
    files = [storage_file, get_fp_file_path(), get_rate_file_path()]

    lock = Lock()
    with lock:
        for file in files:
            if os.path.exists(file):
                os.remove(file)
        global INSTANCE_NUMBER
        INSTANCE_NUMBER = None


def get_storage_path():
    dir_name = MULTI_FP_COLLECTION_FOLDER_NAME if is_multi_fp_collection() else STORAGE_FOLDER_NAME
    return os.path.join(os.path.abspath(os.path.curdir), dir_name)


def get_specific_config_folder_for_fp() -> str:
    """
    returns the destined folder path which the fingerprint should be saved into according to what the config is currently
    :return: string of destined folder
    """

    # TODO: implement system to generate folders for fingerprints
    """# make sure that there exists a folder for each configuration
    for folder in os.listdir(os.path.join(current_folder, "../bd-configs")):"""

    # get currently selected config
    with open(CONFIG_FOLDER + "/current_configuration.json", "r") as file:
        config = json.load(file)

    fingerprint_folder = os.getenv("saved_fingerprints_folder") + "/training"

    # return path of corresponding folder
    for folder in os.listdir(fingerprint_folder):
        if folder.endswith(str(config["current_configuration"])):
            return os.path.join(fingerprint_folder, folder)
    raise FileNotFoundError("Could not find config folder in training fp folder")


def get_fp_file_path():
    if not get_instance_number():
        raise RuntimeError("Execution instance unknown! Must initialize storage first.")
    return os.path.join(get_storage_path(), "fp-{}.txt".format(get_instance_number()))


def get_rate_file_path():
    if not get_instance_number():
        raise RuntimeError("Execution instance unknown! Must initialize storage first.")
    return os.path.join(get_storage_path(), "rate-{}.txt".format(get_instance_number()))


def __get_storage():
    storage_path, _ = __get_storage_file_path()
    return TinyDB(storage_path)


def __prepare_storage_file():
    storage_file, storage_folder = __get_storage_file_path()
    with open(storage_file, "w+"):  # create file if not exists and truncate contents if exists
        pass


def __get_storage_file_path():
    if not get_instance_number():
        raise RuntimeError("Execution instance unknown! Must initialize storage first.")
    storage_folder = os.path.join(current_folder, "..", STORAGE_FOLDER_NAME)
    storage_file = os.path.join(storage_folder, "storage-{}.json".format(get_instance_number()))
    return storage_file, storage_folder


def __determine_instance():
    storage_folder = os.path.join(os.path.abspath(os.path.curdir), STORAGE_FOLDER_NAME)
    os.makedirs(storage_folder, exist_ok=True)

    storage_files = [file for file in os.listdir(storage_folder)
                     if file.startswith("storage-") and file.endswith(".json")]
    instances = list(map(lambda file: int(file.split("-")[1][:-5]), storage_files))
    instances.sort()

    global INSTANCE_NUMBER
    min_instance_number = 1  # avoid failing boolean tests
    if len(instances) > 0:
        # check up to last instance + 2 to account for range indexing and next possible instance
        missing_instances = [item for item in range(min_instance_number, instances[-1] + 2) if item not in instances]
        INSTANCE_NUMBER = min(missing_instances)
    else:
        INSTANCE_NUMBER = min_instance_number


def __query_key(key):
    # print("Query", key)
    flag = None
    lock = Lock()
    with lock:
        max_retries = 3
        # FIXME: solve race condition by properly implementing file lock;
        #  Currently, concurrent read and write operations sometimes lead to improper JSON format
        #  Solved as of now by just re-reading the file
        for i in range(max_retries):  # retry concurrent read at most 3 time
            success = False
            try:
                flag = __get_storage().get(Query().key == str(key))
                success = True
            except Exception as e:
                print("ERROR GETTING STORAGE OF {}, RETRYING {}; ERROR {}".format(key, i + 1, e))
                storage_path, _ = __get_storage_file_path()
                with open(storage_path, "r") as st_f:
                    content = st_f.read()
                print("STORAGE CONTENT:", repr(content))
                if i == max_retries - 1:  # final iteration, zero-based
                    raise e
                if content.endswith('"}}}}'):
                    content = content[:-1].rstrip('\x00')  # Remove null characters
                    with open(storage_path, "w") as st_f:
                        st_f.write(content)
                        st_f.flush()  # Ensure data is written
                        os.fsync(st_f.fileno())  # Sync to disk

            if success:
                break

    assert flag is not None
    # print("{} is {}".format(key, flag["value"]))
    return flag["value"]


def __set_value(key, value):
    lock = Lock()
    with lock:
        # print("Setting", key, "to", value)
        __get_storage().update(set("value", value), Query().key == str(key))
    # print("Set {} to {}".format(key, value))


def map_to_backdoor_configuration(action: int):
    assert 0 <= action < get_num_configs()
    with open(os.path.join(current_folder, "../bd-configs/config-{act}.json".format(act=action)), "r") as conf_file:
        config = json.loads(conf_file.read())
    return config
