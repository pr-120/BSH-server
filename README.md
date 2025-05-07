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
