#!/usr/bin/bash


# folder path of current file
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# load the folder paths from config file
set -a
source $SCRIPT_DIR/../config/folder_paths.config
set +a


# activate anaconda environment
source $anaconda_location/activate
conda activate py310

# Start the server in the background and save the process ID 
echo -e "\n ** STARTING SERVER **\n\n"
python "$api_server_location/server.py" -c &
SERVER_PID=$!


# cleanup procedure to kill server when script terminates
cleanup() {
	echo "Stopping server..."
	ps aux | grep "python $api_server_location/server.py" | awk '{print $2}' | xargs kill 	
}

# kill server when script finishes
trap cleanup EXIT

# short sleep so that terminal output is generated properly
sleep 3  

# which configurations to run, key describes number of config
configurations=( "normal" 1 2 3 4 5 )


# select configs to gather data for
for config in "${configurations[@]}"; do
	
	# start script to set config on client device
	expect -f "$script_folder/set_config_on_client_device.exp" $config
	# save config on server
	echo "{\"current_configuration\": \"$config\"}" > ../config/current_config.json

	# collect more data of normal workings
	if [ "$config" = "normal" ]; then
		number_of_fingerprints_to_be_made=345600
			
		# starts fingerprinting process on client device
		expect -f "$script_folder/start_fingerprinting.exp" $number_of_fingerprints_to_be_made
		
		# terminates when fp is finished 	
		expect -f "$script_folder/check_fp_finished.exp"
		
	else
		number_of_fingerprints_to_be_made=50000
			
		# starts fingerprinting process on client device
		expect -f "$script_folder/start_fingerprinting.exp" $number_of_fingerprints_to_be_made
		
		# create folder for stolen files to be stored in
		mkdir -p $stolen_files_storage_folder
	
		# start malicious process
		expect -f "$script_folder/start_malicious_process.exp"
		
	fi
			
done
