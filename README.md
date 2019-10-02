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
  <a href="#Approach">Approach</a> •
  <a href="#Strategy">Strategy</a> •
  <a href="#Team">Team</a> •
  <a href="#license">License</a>
</p>

## Key Features

The team's main goal is to develop computational algorithms applied to autonomous robots using techniques of artificial intelligence and computer vision. This code was desenvolver for the ROSI CHALLENGE 2019.

## Instructions

##### 1. Create a catking workspace:
```sh
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
```
##### 2. Clone and download this repository package to your ROS Workspace src folder (../catkin_ws_forros/src) folder with the name rosi_defy:
```sh

$ git clone https://github.com/raphaellmsousa/ForROS rosi_defy_forros

```
##### 3. Change the node permission in the .../catkin_ws/src/rosi_defy_forros/script folder:
```sh
cd ~/catkin_ws/src/rosi_defy_forros/script
chmod +x rosi_forros.py
```
##### 4. Compile the node:
```sh
cd ~/catkin_ws
catkin build
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

###### 7.2 Now, let's running out node:
- `$ cd ~/catkin_ws` # open your catkin workspace
- `$ source devel/setup.bash` # source the path
- `$ roslaunch rosi_defy_forros rosi_joy_forros.launch` # start the Rosi node

###### 7.3 Load the vrep scene and start simulation

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

