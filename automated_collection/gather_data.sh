#!/usr/bin/bash


############# CONFIGURATION ##########################

# load the variable passed to the file
config=$1
tick_port=$2

# folder path of current file
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# load the folder paths and app data from config files
set -a
source $SCRIPT_DIR/../config/folder_paths.config
set +a

######################################################



# set config
source $anaconda_location/activate py310
python $api_server_location/set_config.py $config

# collect more data under normal conditions
if [ "$config" = "normal" ]; then
	number_of_fingerprints_to_be_made=10000
			
	# starts fingerprinting process on client device
	expect -f "$script_folder/start_fingerprinting.exp" $number_of_fingerprints_to_be_made $tick_port
	
	# wait until fingerprinting ends and script is reset
	while true; do
		sleep 600
	done
		
else
	number_of_fingerprints_to_be_made=15
			
	# starts fingerprinting process on client device
	expect -f "$script_folder/start_fingerprinting.exp" $number_of_fingerprints_to_be_made $tick_port
		
	# create folder for stolen files to be stored in
	mkdir -p $stolen_files_storage_folder
	
	# start malicious process
	expect -f "$script_folder/start_malicious_process.exp" $tick_port
		
fi
			

