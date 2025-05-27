import json
import os
import socket

current_folder = os.path.dirname(os.path.abspath(__file__))
CONFIG_FOLDER = os.path.join(current_folder, "../../config")

def send_config(num_config, config, ip_address_of_client_device):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip_address_of_client_device, 42666))
        sock.send(bytes(json.dumps(config), encoding="utf-8"))
        print("Sent config", num_config, config)


def save_config_locally(config):
    # save new config locally
    with open(os.path.join(CONFIG_FOLDER, "current_configuration.json"), "w") as f:
        json.dump({"current_configuration": config}, f)

