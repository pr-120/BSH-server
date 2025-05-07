

# Installation

The application expects a working anaconda version installed on the server system. For the application to be able to access anaconda the path must be adjusted in `config/folder_paths.config`. 

There are two anaconda environments used in the application, both must be installed before use. One is located in the `server` folder and the other in the `thetick` folder. In these folders a .yml file is located with which the environment can be generated. To do this use `conda env create -f $name_of_file.yml`.


