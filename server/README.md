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
- `bd-configs/`\
  contains all available backdoor configurations an RL agent can choose from. Taken actions are converted to
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

- **Disclaimer:** _All development regarding the server was run and tested with Python 3.10! Compatibility with other
  Python versions is not guaranteed._

## Prototypes

This table contains high-level summaries of the prototype versions for the RL agents used in this thesis. Not all
prototypes contained in the repository are relevant or were used in the context of this thesis.
For details on the other prototypes, please refer to the previous works mentioned in the thesis.

| Prototype | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 8         | One of the most advanced prototypes of the previous work. Fixes problems such as the "dying ReLU" problem and simulates attack locally to increase realism. Differs from original implementation by using a regular AD system instead of an ideal AD implementation. Prototype can be evaluated in simulation (offline) and based on fingerprints received directly from a target device (online). The other prototypes would theoretically also be able to support online training, but possibly occurring bugs (most certainly) have not been addressed for them. |
| 20        | DDQL with Normal AD Behavior: Implements the Double Deep Q-Learning (DDQL) algorithm while utilizing the anomaly detection (AD) system in its standard form.                                                                                                                                                                                                                                                                                                                                                                                                        |
| 21        | Extends version 20 by introducing a second hidden layer in the Neural network. Differs from original implementation by using a regular AD system instead of an ideal AD implementation.                                                                                                                                                                                                                                                                                                                                                                             |
| 24        | PPO Algorithm with Normal AD Behavior: Introduces the Proximal Policy Optimization (PPO) reinforcement learning algorithm, using the standard AD system to evaluate actions.                                                                                                                                                                                                                                                                                                                                                                                        |


## Auxiliary Scripts

Additionally, there are some auxiliary scripts used for everything around the C2 server.
If a script does not require any parameters or flags, it may also be run in an IDE of your choice for your convenience.
Furthermore, most of the auxiliary scripts use dashes (`-`) instead of underscores (`_`) to avoid any illegal imports in
other scripts since dashes cannot be parsed in import declarations.


### Compare Agent Accuracy

When verification is required that the agent is really learning how to properly select good actions, this script is what
you want to run.

It first creates a fresh instance of an untrained agent and feeds all collected fingerprints for all available
backdoor configs and normal behavior through its prediction.
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

To evaluate the quality of the collected fingerprints, or their underlying backdoor configuration, respectively, we
can pass the collected fingerprints through AD.
This will first evaluate anomaly detection over all previously collected fingerprints (CSV datasets) once with
SimplePreprocessor and once with the CorrelationPreprocessor (highly correlated features are removed).
Finally, the script will evaluate all collected fingerprints that still reside in the collection folder.
This is especially helpful during collection as to detect unexpected behavior or results as early as possible.

Run the script as follows: `python3 evaluate_AD.py`

### Select Evaluation Set

The fingerprints collected
