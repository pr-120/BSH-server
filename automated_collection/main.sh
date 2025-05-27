#!/bin/bash


############# CONFIGURATION ##########################

# load the variable passed to the file
nr_of_devices=${1:-1}

# folder path of current file
CURRENT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# load the folder paths from config file
set -a
source $CURRENT_DIR/../config/folder_paths.config
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
	

      # Calculate ports based on nr_of_devices (starting from 5555)
      ports=()

      for ((device_nr=1; device_nr<=nr_of_devices; device_nr++)); do
              port=$((5555 + (device_nr - 1)))
              ports+=("$port")
              rm -f "$CURRENT_DIR/../config/ls_result_$device_nr.txt"
      done

      # Call terminate_screens.sh with all ports
      if [ ${#ports[@]} -gt 0 ]; then
          bash "$CURRENT_DIR/terminate_screens.sh" "${ports[@]}"
      else
          echo "No devices specified, no SCREEN sessions to terminate."
      fi

      echo "Stopping server..."
      ps aux | grep "python $api_server_location/server.py" | awk '{print $2}' | xargs kill 2>/dev/null

}

# kill server when script finishes
trap cleanup EXIT

# short sleep so that terminal output is generated properly
sleep 3  

# which configurations to run, key describes number of config
configurations=( 1 2 "normal" )


# select configs to gather data for
for config in "${configurations[@]}"; do

    # save config on server
    echo "{\"current_configuration\": \"$config\"}" > $CURRENT_DIR/../config/current_configuration.json

    printf "\nConfiguration $config is being run:\n"
    for ((device_nr=1; device_nr<=nr_of_devices; device_nr++)); do
	
        # we assign each device a single port to communicate over (5555 is used as this is the standard port)
        port=$((5555 + (device_nr - 1)))

        # start screen session in background
        screen -dmS "tick_$port" -L -Logfile "$CURRENT_DIR/LOGFILES/LOGFILE_$port.txt" bash "$CURR_DIR/gather_data.sh" "$config" "$port"

    done

    # Wait until all SCREEN sessions finish
    while ps aux | grep -P "SCREEN -dmS tick_\d+" >/dev/null 2>&1; do
        printf "\ndata being collected\n"
	      sleep 15
    done
	
done

