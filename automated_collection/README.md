# AUTOMATED COLLECTION

Scripts located in this directory are used to automatically collect behavioral fingerprint data from Raspberry Pis.

## Important files

- `main.sh` is the starting point of the scripts. This script takes the number of devices used to collect the data as an
  argument. The configurations for which data is supposed to be collected are defined in a list.

- `gather_data.sh` starts the fingerprinting process and the execution of configurations on the client device. The
  script
  invokes a screen instance for every device collecting data. These screens are terminated when the client finishes with
  fingerprinting and signals this to the API. The number of fingerprints to be made for normal and malicious
  behaviors are defined  here.

- `start_malicious_process.exp` starts the execution of the malicious behavior of the backdoor. Executes a recursive
  procedure which starts in the `/home` directory and transfers all subfolders to the C&C server.

## Configuration

To be able to set the configurations on the client devices, the IP address is required of all devices. These must be
given in the `server/environment/settings.py` file. 

    IP_DEVICE_5555 = "192.168.191.242"
    IP_DEVICE_5556 = "192.168.191.212"
    # add as many devices as needed ...
    
    CLIENT_DEVICES = [IP_DEVICE_5555, IP_DEVICE_5556]
