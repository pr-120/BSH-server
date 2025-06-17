# API (Server)

## Structure

There are some components that are used globally and other components that are specific to a particular version of a
reinforcement learning (RL) agent prototype.

The globally used components are stored in their respective package:

- `agent/`\
  contains the `AbstractAgent` class and all files related to selecting and constructing an `Agent`, be it from a fresh
  instance or from a representation files obtained through training an agent.
- `api/`\
  contains the Flask app to start the command and control (C&C) server API. The various endpoints are split semantically
  into corresponding files, e.g., everything to do with receiving fingerprints and encryption rates is stored in the
  `fingerprint.py` file.
- `environment/`\
  contains the main parts of the RL environment, such as the `AbstractController` (orchestration of an `Agent` and its
  training process), `AbstractPreprocessor` (preprocessing fingerprints for anomaly detection (AD) or passing through a
  neural network in the `Agent`), or `AbstractReward` (computing rewards for states/actions based on the results of AD).
  Moreover, the main settings file and methods to handle the storage of a single run to allow parallel executions are
  also contained in this folder.
- `rw-configs/`\
  contains all available ransomware configurations an RL agent can choose from. Taken actions are converted to
  configurations that are then sent to the ROAR client for integration into the encryption process.
- `utilities/`\
  contains all sorts of scripts used throughout the framework. These scripts include handler methods to write received
  metrics into files for inter-process communication, plotting of episode metrics like received rewards or the number of
  steps, and helper methods to simulate the environment without relying on the API.

Some folders are not part of the GitHub repository because they are dynamically created and filled with content based on
your usage of the server.
These folders include the `fingerprints` folder, used for saving received fingerprints during collection mode, and the
`storage` folder, used for storing all files belonging to a particular run, i.e., storage files, rate files, state (
fingerprint) files, results, and plots.

The prototype-specific components are stored in a folder of their respective prototype version, i.e., `vX/` for
prototype version `X`.
In there you can find all files that overwrite certain behavior or are otherwise specific to this prototype.
The components are arranged the same way the global components are arranged, for example, the prototype-specific
implementation of the `ControllerOptimized` for prototype version 8 is stored in the `v8/environment` package.

## Prototypes

This table contains high-level summaries of the prototype versions for an RL agent used in this thesis. Not all
prototypes contained in the repository are relevant or were used in the context of this thesis.
For details on the other prototypes, please refer to the previous works mentioned in the thesis.

| Prototype | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 8         | One of the most advanced prototypes of the previous work. Fixes problems such as the "dying ReLU" problem and simulates attack locally to increase realism. Differs from original implementation by using a regular AD system instead of an ideal AD implementation. Prototype can be evaluated in simulation (offline) and based on fingerprints received directly from a target device (online). The other prototypes would theoretically also be able to support online training, but possibly occurring bugs (most certainly) have not been addressed for them. |
| 20        | DDQL with Normal AD Behavior: Implements the Double Deep Q-Learning (DDQL) algorithm while utilizing the anomaly detection (AD) system in its standard form.                                                                                                                                                                                                                                                                                                                                                                                                        |
| 21        | Extends version 20 by introducing a second hidden layer in the Neural network. Differs from original implementation by using a regular AD system instead of an ideal AD implementation.                                                                                                                                                                                                                                                                                                                                                                             |
| 24        | PPO Algorithm with Normal AD Behavior: Introduces the Proximal Policy Optimization (PPO) reinforcement learning algorithm, using the standard AD system to evaluate actions.                                                                                                                                                                                                                                                                                                                                                                                        |


## Run the Server

There are different ways to use the `server.py` according to the intended purpose.

- **Disclaimer:** _All development regarding the server was run and tested with Python 3.10! Compatibility with other
  Python versions is not guaranteed._

### Collect Fingerprints

Fitting if the intention is to only listen to the API for collection of incoming fingerprints.
Collected fingerprints are stored in the [fingerprints folder](./fingerprints), specifically in the
subfolder [training](./fingerprints/training). To train and evaluate the different components of the application, an
evaluation set must be created. This can be done by using [move_files()](./select-fingerprints-test-set.py).
The subfolders in [Folder Structure](#folder-structure).

Start the server as follows: `python3 server.py -c`

### Train Agent (Online on Target Device)

The agent can be trained in an online step-by-step fashion with fingerprints directly originating from the target
device.
This is the most realistic scenario, but it trades higher authenticity for slower execution as the fingerprints
collected on the device arrive roughly every 7 seconds.
Accordingly, the time between every step the agent can take is quite high, and learning is very slow - independent of
the underlying implementation.

Start the server as follows: `python3 server.py -p <x>` (`<x>` represents the selected version number)

### Train Agent (Offline Simulation)

To tackle the problem of slow learning and dependency on a target device, the agent can be trained in a simulation with
fingerprints originating from a corpus of previously collected fingerprints.
This allows to invest the time for collecting fingerprints only once and then run many simulations on the collected
data.
While this comes with a much faster execution, it trades speed for authenticity because the environment is entirely
artificial and may not represent the actual environment on the target device.

Start the server as follows: `python3 server.py -p <x> -s`

### Available Options

- `-c | --collect`\
  As mentioned in the scenario "Collect Fingerprints", this flag marks collection of fingerprints and does not start the
  RL agent.
- `-p | --proto`\
  As mentioned in the scenario "Train Agent (Online on Target Device)", this flag configures the prototype version to be
  used.
  It takes an integer as argument (should correspond to a valid prototype version) and stores it in the storage file.
  This configuration impacts the
  chosen [Controller](./environment/abstract_controller.py), [Agent](./agent/abstract_agent.py), and corresponding
  Model.
  Accordingly, it also impacts the applied fingerprint [Preprocessor](./environment/anomaly_detection) and the
  used [Reward](./environment/reward) system.
- `-r | --representation`\
  Any scenario typically creates a new instance of the configured agent class.
  Providing this flag initiates a different instantiation of the agent.
  It takes an absolute path to an [AgentRepresentation](./agent/agent_representation.py) file as an argument
  and [initiates the agent](./agent/constructor.py)'s internal properties based on this representation.
  Therefore, providing a pretrained agent allows breaking down long training periods into smaller periods.
- `-s | --simulation`\
  As mentioned in the scenario "Train Agent (Offline Simulation)", this flag marks the execution of the RL agent in
  simulation.
  Therefore, the Flask API is not started and all required API interactions are executed programmatically.
  Also, the fingerprints for state representation are not received from the client but randomly selected from the
  respective training set matching the taken action.

## Auxiliary Scripts

Additionally, there are some auxiliary scripts used for everything around the C2 server.
If a script does not require any parameters or flags, it may also be run in an IDE of your choice for your convenience.
Furthermore, most of the auxiliary scripts use dashes (`-`) instead of underscores (`_`) to avoid any illegal imports in
other scripts since dashes cannot be parsed in import declarations.

- **Disclaimer:** _All development regarding the auxiliary scripts was run and tested with Python 3.10! Compatibility
  with other Python versions is not guaranteed._

### Compare Agent Accuracy

When verification is required that the agent is really learning how to properly select good actions, this script is what
you want to run.

It first creates a fresh instance of an untrained agent and feeds all collected fingerprints for all available
ransomware configs and normal behavior through its prediction.
The expected result is very bad as the agent's initial prediction skills are more or less equivalent to random guessing.
Then, the untrained agent is trained according to the current environment settings.
Lastly, the now trained agent is again evaluated by feeding all fingerprints through its prediction.
This time, however, the expected results are much better than before and clearly show that the agent is no longer
guessing but demonstrates being able to select good actions for all possible states.

To avoid unwanted influences, the evaluation of agent performance is done using a dedicated evaluation set of
fingerprints instead of the regular training set of fingerprints.
In addition, during both evaluation phases, the agent is only predicting actions but not learning from its choices, such
that the evaluation set can still be considered "never seen before".

Run the script as follows: `python3 NAME_OF_SCRIPT.py`
For different Implementations, different Scripts need to be used.

| Prototype | Accuracy Script     |
|------|---------------------|
| 8    | `accuracy.py`       |
| 20   | `accuracy.py`       | 
| 21   | `accuracy_DDQL.py`  |
| 24   | `accuracy_PPO.py`   |

### Convert Fingerprints to CSV Files

Run this script to convert the collected fingerprints from the target device to a CSV file for further usage.
The respective target set of fingerprints (training or evaluation) must be configured at the top of the script.

Run the script as follows: `python3 fp-to-csv.py`

### Evaluate Anomaly Detection (AD)

To evaluate the quality of the collected fingerprints, or their underlying ransomware configuration, respectively, we
can pass the collected fingerprints through AD.
This will first evaluate anomaly detection over all previously collected fingerprints (CSV datasets) once with
SimplePreprocessor and once with the CorrelationPreprocessor (highly correlated features are removed).
Finally, the script will evaluate all collected fingerprints that still reside in the collection folder.
This is especially helpful during collection as to detect unexpected behavior or results as early as possible.

Run the script as follows: `python3 evaluate_AD.py`

### Find Average Encryption Rate of Configurations

The environment simulation relies on the encryption rate of every ransomware configuration to compute the performance
reward.
However, no fixed value is available for configurations with unlimited encryption rate.
Accordingly, manually determining and configuring the average encryption rate for ransomware configurations with
unlimited rates is required.
This is what this script is for:
it iterates over all collected encryption metrics for each configuration, extracts the reported encryption rate, and
computes the average.
The reported average rate can then be manually configured in the [simulation file](./utilities/simulate.py).

Run the script as follows: `python3 find_avg_rate.py`

### Plot Activation Function

Pretty much what the title says:
adjust the script to match two activation functions with all corresponding parameters and plot them together for
comparison.
This script was mainly used to generate the figures (different combinations of activation functions) that were later
included in the thesis report.

Run the script as follows: `python3 plot_activation_func.py`

### Plot Reward Function

Pretty much what the title says:
adjust the script to match two reward functions with all corresponding parameters and plot them together for comparison.
This script was mainly used to generate the figures (different versions of reward computation) that were later included
in the thesis report.

Run the script as follows: `python3 plot_perf_reward_func.py`
