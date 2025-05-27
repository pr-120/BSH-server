#!/bin/bash


############# CONFIGURATION ##########################

# load the variable passed to the file
ip_of_training_device=$1

# folder path of current file
CURR_DIR="$(dirname "${BASH_SOURCE[0]}")"

# load the folder paths from config file
set -a
source $CURR_DIR/config/folder_paths.config
set +a

######################################################




# cleanup procedure to kill server when script terminates
cleanup() {
	
	# kill remote shell after finishing
	bash "$script_folder/terminate_screens.sh" "5555"
    	
	echo "Stopping server..."
	ps aux | grep "python $api_server_location/server.py" | awk '{print $2}' | xargs kill 2>/dev/null 	

}
# kill server when script finishes
trap cleanup EXIT


tick_port="5555" 


# starts infinite fingerprinting process on client device
expect -f "$script_folder/start_fingerprinting.exp" 0 $tick_port

# activate anaconda environment
source $anaconda_location/activate py310
	
# set default config on client device
python $api_server_location/set_config.py 0 192.168.191.242

# create folder for stolen files to be stored in
mkdir -p $stolen_files_storage_folder


# Start the server in the background and save the process ID 
echo -e "\n ** STARTING SERVER **\n\n"
python "$api_server_location/server.py" -p "8" &
SERVER_PID=$!

	
# start malicious process
expect -f "$script_folder/start_malicious_process.exp" $tick_port

		
