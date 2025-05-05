# Setup ROAR Environment

This file contains instructions on how to set up the C&C server to get it running.
Additionally, some auxiliary scripts are available for advanced usage concerning data analytics or extension and configuration of the C&C server.





## Install Python and Dependencies
First, make sure you have a compatible Python version installed on your system.
All development regarding the server was run and tested with Python 3.10.
Compatibility with other Python versions is not guaranteed.

Second, you need to install the dependencies required by this repository.
As with any Python repository, you'll need pip for that (use the specific version belonging to the Python version you use, e.g., pip 22.3 for Python 3.10).
The list of dependencies is available in [requirements.txt](./requirements.txt).
In order to install, run the following command in the root directory of this repository: `pip install -r requirements.txt`





## Configuration
To simply launch the server, adjust constants in the files listed below.
The settings file is the main configuration file containing only global constants.
The constants of all other files are typically located at the top of the file, unless indicated otherwise.
If nothing is indicated and there are constants further down in the file, it may be advisable not to change them.
Handle with care and at your own risk of breaking stuff! 

| File                                                 | Constant                                                                                                       |
|------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `environment/anomaly_detection/anomaly-detection.py` | - Contamination factor                                                                                         |
| `environment/settings.py`                            | - CSV folder path<br>- Verify CSV headers<br>- Client IP address<br>- AD features<br>- C&C simulation settings |
| `vX/agent/agent.py`                                  | - Agent specific constants and starting values ()                                                              |
| `vX/agent/model.py`                                  | - Model specific constants and starting values                                                                 |
| `vX/environment/controller.py`                       | - Episode specific constants and values (orchestration)                                                        |

To be able to used auxiliary scripts, there are other constants contained in the files listed below that may require adjustments - depending on the script to be run.

| File                                                | Constant                                                                                                       |
|-----------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `environment/reward/ideal_AD_performance_reward.py` | - Detected configurations (only check if original set was altered)                                             |
| `environment/abstract_controller.py`                | - Wait for user confirmation (default False)                                                                   |
| `environment/state_handling.py`                     | - Folder names (DANGER ZONE!)                                                                                  |
| `environment/settings.py`                           | - CSV folder path<br>- Verify CSV headers<br>- Client IP address<br>- AD features<br>- C&C simulation settings |
| `utilities/simulate.py`                             | - Configuration settings for unlimited encryption rate                                                         |
| `accuracy.py`                                       | - Follow instructions at the top                                                                               |
| `accuracy_pretrained.py`                            | - Follow instructions at the top                                                                               |
| `find-avg-rate.py`                                  | - Metrics folder path (default training set folder)                                                            |
| `fp-to-csv.py`                                      | - CSV file names<br>- Verify CSV headers                                                                       |
| `plot-activation-func.py`                           | - Activation functions<br>- Plot axis range and descriptions                                                   |
| `plot-perf-reward-func.py`                          | - Reward functions<br>- Plot axis range and descriptions                                                       |
| `select-fingerprints-test-set.py`                   | - Fingerprint folders (source and target)                                                                      |

Other files should technically not require any changes to configurations as they import the requirements from the settings file.
But then again, some files or configurations may have been overlooked.
So best to try first if it works out of the box or after the adjustments listed above.
If not, then have a go at debugging and changing values but watch out for side effects due to global usage of some constants.





## Fingerprints Folder Structure

All scripts contained in this repository can only work if the required data can be found, i.e., the collected fingerprints need to be stored in a very specific way.

```
FOLDER                  DESCRIPTION

fingerprints            # The local folder containing all respective subdirectories. This folder and its children are not required to be located in this repository as long as the corresponding settings are correctly set.
-- evaluation           # The subfolder where the portion of fingerprints explicitly used only in accuracy computation is stored. The corresponding setting is called `EVALUATION_CSV_FOLDER_PATH`.
    -- infected-cX      # Directory for all infected-behavior fingerprints belonging to ransomware configuration X. There should be one folder for every configuration.
    -- normal           # Directory for normal-behavior fingerprints. There should be exactly one such folder here.
-- training             # The subfolder where all other fingerprints used during training will be stored. The corresponding setting is called `TRAINING_CSV_FOLDER_PATH`.
    -- infected-cX      # Directory for all infected-behavior fingerprints belonging to ransomware configuration X. There should be one folder for every configuration.
    -- normal           # Directory for normal-behavior fingerprints. There should be exactly one such folder here.
```





## Setup Complete
All done!
Now you should be able to run the scripts as presented in the [main README](./README.md#Run the Server).
