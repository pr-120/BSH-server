#!/usr/bin/bash


############# CONFIGURATION ##########################

# load the variable passed to the file
config=$1
tick_port=$2

# folder path of current file
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# load the folder paths from config file
set -a
source $SCRIPT_DIR/../config/folder_paths.config
set +a

######################################################



# set config on client device
expect -f "$script_folder/set_config_on_client_device.exp" $config $tick_port


# collect more data under normal conditions
if [ "$config" = "normal" ]; then
	number_of_fingerprints_to_be_made=17500
			
	# starts fingerprinting process on client device
	expect -f "$script_folder/start_fingerprinting.exp" $number_of_fingerprints_to_be_made $tick_port
	
	# wait until fingerprinting ends and script is reset
	while true; do
		sleep 600
	done
		
else
	number_of_fingerprints_to_be_made=1500
			
	# starts fingerprinting process on client device
	expect -f "$script_folder/start_fingerprinting.exp" $number_of_fingerprints_to_be_made $tick_port
		
	# create folder for stolen files to be stored in
	mkdir -p $stolen_files_storage_folder
	
	# start malicious process
	expect -f "$script_folder/start_malicious_process.exp" $tick_port
		
fi
			

