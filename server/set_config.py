from api.configurations import send_config, save_config_locally
from environment.settings import CLIENT_DEVICES
from environment.state_handling import map_to_backdoor_configuration
import sys


def main():
    save_config_locally(sys.argv[1])

    # if benign behavior is being tested the config doens't need to be set on the victim device
    if sys.argv[1] == "normal":
        return

    config_nr = int(sys.argv[1])

    config_mapping = map_to_backdoor_configuration(config_nr)

    for client in CLIENT_DEVICES:
        try:
            send_config(config_nr, config_mapping, client)
        except ConnectionRefusedError:
            print(f"{client}: Connection refused.")

if __name__ == "__main__":
    main()
