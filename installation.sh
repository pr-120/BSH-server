#!/bin/bash

############# CONFIGURATION ##########################

# folder path of current file
CURR_DIR="$(dirname "${BASH_SOURCE[0]}")"

# load the folder paths from config file
set -a
source $CURR_DIR/config/folder_paths.config
set +a

######################################################


# install conda environments
cd $tick_remote_shell_folder
$anaconda_location/conda env create -f tick_environment.yml

cd $api_server_location
$anaconda_location/conda env create -f api_environment.yml


# install necessary packages
additional_packages=( "expect" "screen" )
apt-get update
for package in "${additional_packages[@]}"; do
	 apt-get install -y "$package"
done
