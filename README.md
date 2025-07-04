# BSH-server

Bachelor thesis on Backdoor Attacks Optimized with RL for Resource-constrained devices.
The official title of this thesis is *AI-powered Backdoor to Stay Hidden*. (Referred to as **BSH** from here on out)

This repository contains the RL Agent and command and control (C&C) part of the project. There is [another
repository](https://github.com/pr-120/BSH-client) for the client device software (backdoor, fingerprint collection,
additional behaviors, etc.)

Note: This README only covers the extensions made, for the other parts refer to the previous work.

## Configuration

The `folder_paths.config` file defines where the application expects the folders to be. The file is
split into two parts, the paths of folders on the server side and on the client side. Be aware that these paths must be
adjusted should there be any changes to the structure of server or client side.

To set up the client device(s) the IPs must be configured in `server/environment/settings.py`. Each device must be added 
to the CLIENT_DEVICES variable. The live training device is selected with the LIVE_TRAINING_DEVICE variable.

```
IP_DEVICE_5555 = "YOUR IP"
# add as many devices as needed ...

CLIENT_DEVICES = [IP_DEVICE_5555]

LIVE_TRAINING_DEVICE = IP_DEVICE_5555
```

## Setup

The `installation.sh` script manages a clean install. For the script to work there must be a few things given.
The program was developed for unix systems. The system used Ubuntu 24.04.2 LTS, therefore *apt-get* was used to install
packages on the device. When using other distributions and/or other package managers, changes need to be made for the
installation script to work properly.

The application expects a working anaconda/miniconda version installed on the server system. For the application to be able to access anaconda the path must be adjusted in `config/folder_paths.config`. The application was developed
using anaconda version 24.9.2. Compatability with other versions is not guaranteed.


## Structure

There are some components that are used globally and other components that are specific to a particular version of a
reinforcement learning (RL) agent prototype.

The globally used components are stored in their respective package:

- `server/`\
  contains the code for the API and the model training. This is where most code reused from previous work is located in.
  There is an additional README file giving context on this part of the application.
- `thetick`\
  contains the code for the backdoor remote console. Also contains corresponding yml file for versioning and packaging.
- `automated_collection`\
  contains scripts used to help automate the collection of data for the training of the RL-agents.
- `stolen_files`\
  stores the files which have been exfiltrated from client devices. Contains subfolders with the schema
  `$port_of_tick_console`, named after the individual ports the remote shells were active on and collected the data
  from.
- `config`\
  contains files which are used to provide context throughout the application.
- `__data`\
  contains all the fingerprint data collected and used in this thesis, as well as the evaluation results and plots.
- `LICENSES`\
  contains the licensing files (multiple) for this application.


## Run

### Data collection:

To run the data collection the `main.sh` file needs to be executed in the `automated_collection` folder. This file
requires the number of devices which are used to record the fingerprints as a parameter.

The application starts that many remote shell instances, all assigned to different ports. This means that the client
devices need to be configured to individual ports as well (in the `app_data.config` on the client device). The default
port is 5555 and with each additional device the
port number is increased by one (i.e. 5556 for device 2, 5557 for device 3, etc.).

The remote shells are started as screen sessions which run in the background, making it possible to run multiple
processes at once. The output of the screen sessions is hidden but the screens can be reattached to the
console by using `screen -r $name_of_screen`. All available screens are shown with `screen -ls`.

### Live training of RL-agent:

The `live_train.sh` script collects fingerprints from the client device and trains the RL-agent in real-time. This is
done using only one device on the standard 5555 port. 

Prototype 8 is the only prototype which was specifically tested for live training. Therefore the script automatically runs with this prototype. Other prototypes should work in theory but no guarantee is given.
