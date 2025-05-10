# BSH-server
Server side of BSH application

## Configuration 

The `folder_paths.config` file defines where the application tries to access and store specific files from. The file is split into two parts, the paths of folders on the server side and on the client side. 

The server configuration paths expect the root of the application to be placed in the home directory of the C&C server. If the user wishes to use another location this must be amended in the file.

"""
project_location="$HOME/BSH-server"
"""

This dictates affects all the other folder locations on the server side.

For the folder locations of the client side the root of the application must be known. The absolute path of this location must be given as the remote shell will not be able to translate relative paths correctly on the client device.

ex: ~/BHS-client --will-be-wrongly-translated-to--> $home_of_server_side/BHS-client

## Run

To run the application the `main.sh` file needs to be executed in the `scripts` folder. This file requires the number of devices which are used to record the fingerprints to be passed to it. 

The application starts that many remote shell instances, all assigned to different ports. This means that the client devices need to be configured to individual ports as well. The default port is 5555 and with each additional device the port number is increased by one (i.e. 5556 for device 2, 5557 for device 3, etc.). 

The remote shells are started as screen sessions which run in the background, making it possible to run multiple processes at once. The program runs as long as any screen sessions are still active. After this it continues to measuring the next configuration. The output of the screen sessions is hidden but the screens can be reattached to the console by using `screen -r $name_of_screen`. All available screens are shown with `screen -ls`


