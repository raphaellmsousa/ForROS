 <h1 align="center">
  <br>
  <a href="https://www.overleaf.com"><img src="https://user-images.githubusercontent.com/31168586/65396910-dbd1f300-dd81-11e9-9a98-8f4f329461e0.png" alt="Overleaf" width="400"></a>
</h1>

<h4 align="center">

CHALLENGE ROSI 2019 Competition Team</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="https://github.com/overleaf/overleaf/wiki">Wiki</a> •
  <a href="#Instructions">Instructions</a> •
  <a href="#Approach">Team</a> •
  <a href="#Strategy">Team</a> •
  <a href="#Team">Team</a> •
  <a href="#license">License</a>
</p>

## Key Features

The team's main goal is to develop computational algorithms applied to autonomous robots using techniques of artificial intelligence and computer vision. This code was desenvolver for the ROSI CHALLENGE 2019.

## Instructions

##### 1. First of all, follow all instructions to install vrep simulator for the ROSI Challenge 2019, as you can find in:

https://github.com/filRocha/rosiChallenge-sbai2019

##### 2. Now, download our repositorty and copy the follow files in the "script" folder (in the rosi_defy folder, check this folder in the step 1):
```sh
- rosi_joy.py # Rosi node
- model.h # Trainned CNN model to avoid obstacles
- modelLadder.h # Trainned CNN model to go up the stairs
- model.py # To train a new CNN model
```
##### 3. Now, create the follow folders in the script folder (this is used to create your own dataset):
```sh
- rgb_data # To save data for training a new CNN model
- robotCommands # To sabe the xls file with the motors traction commands
- map # To save the GPS tracking
```
##### 4. Replace your own paths in the Rosi node (rosi_joy.py file) 
Obs.: open your script folder by using the bash and type the follow command to get the right path:
```sh
$ pwd
```

So, use this path to replace thease lines in the rosi_joy.py file
```sh
- model1 = load_model('/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/model.h5') 
- model2 = load_model('/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/modelLadder.h5') 
- path_to_imageName1 = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/'+imageName1+'/' 
- path_to_imageName2 = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/'+imageName2+'/' 
- path_to_folder = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/robotCommands/' 
- path_to_folder = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/map/' 
- plt.savefig('/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/map/map.png') 
```
##### 5. Install the dependences:
Obs.: as we are using python 2.7, you must use pip2 to install the follow dependences.
```sh
- $ sudo apt install python-pip # pip2 install
- $ pip2 install "numpy<1.17" # Numpy version<1.17
- $ pip2 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.14.0-cp27-none-linux_x86_64.whl # Tensorflow version 1.14.0
- $ pip2 install keras==2.2.5 # Keras version 2.2.5
```
##### 6. Config the "simulation_parameters.yaml" as bellow:

```sh
 # rosi simulation parameters
 rosi_simulation: 
 
  # Simulation Rendering Flag
  # disable it for faster simulation (but no visualization)
  # possible values: true, false.
  simulation_rendering: true

  # Velodyne processing flag
  # disable it for faster simulation (but no Velodyne data)
  # possible values: true, false.
  velodyne_processing: false

  # Kinect processing flag
  # disable it for faster simulation (but no kinect data)
  # possible values: true, false.
  kinect_processing: true

  # Hokuyo processing flag
  # disable it for faster simulation (but no hokuyo data)
  # possible values: true, false.
  hokuyo_processing: false

  # Hokuyo drawing lines 
  # enable it for seeing hokuyo lines
  # possible values: true, false.
  hokuyo_lines: false

  # Camera on UR5 tool processing flag
  # enable it for processing/publishing the ur5 tool cam
  # possible values: true, false
  ur5toolCam_processing: true

  # Fire enabling
  # allows to enable/disable the fire
  fire_rendering: true
```

##### 7. Open a bash terminal and type the follow commands:
- `$ roscore` # start a ROS master

###### 7.1 In a new bash tab:
- `$ vrep` # to open the vrep simulator

###### 7.2 In a new bash tab:
- `$ cd catkin_ws` # open your catkin workspace
- `$ source deve/setup.bash` # source the path
- `$ roslaunch rosi_defy rosi_joy.launch --screen` # start the Rosi node

##### 8. Load the vrep scene and start simulation

## Approach

A detailed description of our team's approach has been provided in the jupyter notebook file "valeNeuralNetwork.ipynb"

## Strategy

Our team divided the mission into two laps. The first one, we turn around the treadmills avoiding obstacles and passing through the restricted region. Also, we are detecting fire while the robot is running, our GPS movement and the position of detected fire is presented in a realtime map, check the "valeNeuralNetwork.ipynb" for more details. 

In the second lap, our robot is going to try climbing up the ladders and touch the rolls. We are going to touch first the suspended platform rolls on fire and after down the stairs, the robot will detect and touch the base of a roll without fire. After that, our mission is concluded.

## Team

Institutions: Federal Institute of Paraiba - IFPB - (Cajazeiras) and Federal Institute of Bahia - IFBA (Vitoria da Conquista).
* Raphaell Maciel de Sousa (team leader/IFPB)
* Gerberson Felix da Silva (IFPB)	
* Jean Carlos Palácio Santos (IFBA)
* Rafael Silva Nogueira Pacheco (IFBA)
* Michael Botelho Santana (IFBA)
* Sérgio Ricardo Ferreira Andrade Júnior (IFBA)
* Lucas dos Santos Ribeiro (IFBA)
* Félix Santana Brito (IFBA)
* José Alberto Diaz Amado (IFBA)


## License

Copyright (c) ForROS, 2019.
