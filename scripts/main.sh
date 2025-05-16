#!/bin/bash


############# CONFIGURATION ##########################

# load the variable passed to the file
nr_of_devices=$1

# folder path of current file
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# load the folder paths from config file
set -a
source $SCRIPT_DIR/../config/folder_paths.config
set +a

######################################################



# activate anaconda environment
source $anaconda_location/activate py310

# Start the server in the background and save the process ID 
echo -e "\n ** STARTING SERVER **\n\n"
python "$api_server_location/server.py" -c &
SERVER_PID=$!


# cleanup procedure to kill server when script terminates
cleanup() {
	
	# kill remote shells as well
	bash $SCRIPT_DIR/terminate_screens.sh

	echo "Stopping server..."
	ps aux | grep "python $api_server_location/server.py" | awk '{print $2}' | xargs kill 2>/dev/null 	

}

# kill server when script finishes
trap cleanup EXIT

# short sleep so that terminal output is generated properly
sleep 3  

# which configurations to run, key describes number of config
configurations=( "normal" 1 2 3 4 5 )


# select configs to gather data for
for config in "${configurations[@]}"; do

    # save config on server
    echo "{\"current_configuration\": \"$config\"}" > ../config/current_config.json

    printf "\nConfiguration $config is being run:\n"
    for ((device_id=1; device_id<=nr_of_devices; device_id++)); do
	
	# we assign each device a single port to communicate over (5555 is used as this is the standard port)
	port=$((5555 + (device_id - 1)))	

	# start screen session in background
	screen -dmS "tick_$device_id" bash "$SCRIPT_DIR/gather_data.sh" "$config" "$port"

    done

    # Wait until all SCREEN sessions finish
    while ps aux | grep -P "SCREEN -dmS tick_\d+ bash $SCRIPT_DIR/gather_data.sh" >/dev/null 2>&1; do
        printf "\ndata being collected\n"
	sleep 15
    done
	
done

