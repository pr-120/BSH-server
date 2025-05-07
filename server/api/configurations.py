import json
import os
import socket

from environment.settings import CLIENT_IP
from environment.state_handling import get_num_configs


def send_config(num_config, config):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((CLIENT_IP, 42666))
        sock.send(bytes(json.dumps(config), encoding="utf-8"))
        print("Sent config", num_config, config)


def map_to_ransomware_configuration(action):
    assert 0 <= action < get_num_configs()
    with open(os.path.join(os.path.curdir, "./bd-configs/config-{act}.json".format(act=action)), "r") as conf_file:
        config = json.loads(conf_file.read())
    return config
