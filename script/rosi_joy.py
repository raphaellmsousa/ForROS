#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

###############################################################################################################################
#			
#	This code has been developed for the ROSI Challenge 2019 https://github.com/filRocha/rosiChallenge-sbai2019
#	Team: ForROS		
#	Institutions: Federal Institute of Paraiba (Cajazeiras) and Federal Institute of Bahia	
#	Team: Raphaell Maciel de Sousa (team leader/IFPB)
#		Gerberson Felix da Silva (IFPB)	
#		Jean Carlos Palácio Santos (IFBA)
#		Rafael Silva Nogueira Pacheco (IFBA)
#		Michael Botelho Santana (IFBA)
#		Sérgio Ricardo Ferreira Andrade Júnior (IFBA)
#		Matheus Vilela Novaes (IFBA)		
#		Lucas dos Santos Ribeiro (IFBA)
#		Félix Santana Brito (IFBA)
#		José Alberto Diaz Amado (IFBA)
#
#	Approach: it was used a behavioral clonning technique to move the robot around the path and avoid obstacles.
#
###############################################################################################################################

import rospy
import numpy as np
from rosi_defy.msg import RosiMovement
from rosi_defy.msg import RosiMovementArray
from rosi_defy.msg import ManipulatorJoints
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import NavSatFix

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from sensor_msgs.msg import PointCloud
from rosi_defy.msg import HokuyoReading

import os
import os.path

import csv

from keras.models import load_model

import matplotlib.pyplot as plt

###############################################################################################################################
#	Uncomment to use the model
###############################################################################################################################

model1 = load_model('/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/model.h5')
model1._make_predict_function()

model2 = load_model('/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/modelLadder.h5')
model2._make_predict_function()

###############################################################################################################################
#	Instructions
###############################################################################################################################

# Press button 10 (depends on your control configuration) to record data and training the (Convolutional Neural Network) CNN
# Press button 9 (keep pressing to running the model)
# Tensorflow version 1.14.0
# Keras version 2.2.5

class RosiNodeClass():

	# class attributes
	max_translational_speed = 5*20 # in [m/s]
	max_rotational_speed = 10*4 # in [rad/s]
	max_arms_rotational_speed = 0.52 # in [rad/s]

	# how to obtain these values? see Mandow et al. COMPLETE THIS REFERENCE
	var_lambda = 0.965
	wheel_radius = 0.1324
	ycir = 0.531

	# class constructor
	def __init__(self):

		#rospy.loginfo("This is a test!")		

		# initializing some attributes
		self.omega_left = 0
		self.omega_right = 0
		self.arm_front_rotSpeed = 0
		self.arm_rear_rotSpeed = 0
		self.camera_image = None
		self.camera_image_depth = None
		self.bridge = CvBridge()
		self.velodyneOut = None
		self.hokuyoOut = None
		self.robotMovement = None
		self.tractionCommand = None
		self.countImageDepth = 0
		self.countImageRGB = 0
		self.flag = "w"
		self.velodyne = None
		self.save_image_flag = False #Flag to save image
		self.steering_angle = None
		self.autoModeStart = True #Default False
		self.countHokuyo = 0
		self.mediaVector = 0
		self.moveArm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		self.camera_image_arm = None
		self.img_out_arm = None
		self.rgbOut = None
		self.concatImage = None
		self.contStart = 0
		self.offset = 0
		self.moveJointLeft = 0
		self.moveJointRight = 0
		self.thetaAll = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # in deg
		self.selectFunction = 0
		self.jointSelect = 0
		self.thetaJoint = 0
		self.resetArm = 0
		self.latitudes = []
		self.longitudes = []
		self.gpsInc = 0
		self.latitude = 0
		self.longitude = 0
		self.autoMode = 0
		self.trigger_right = 0
		self.trigger_left = 0

		# computing the kinematic A matrix
		self.kin_matrix_A = self.compute_kinematicAMatrix(self.var_lambda, self.wheel_radius, self.ycir)

		# sends a message to the user
		rospy.loginfo('Rosi_joy node started')

		# registering to publishers
		self.pub_traction = rospy.Publisher('/rosi/command_traction_speed', RosiMovementArray, queue_size=1)
		self.pub_arm = rospy.Publisher('/rosi/command_arms_speed', RosiMovementArray, queue_size=1)
		self.pub_kinect_joint = rospy.Publisher('/rosi/command_kinect_joint', Float32, queue_size=1)
		self.pub_jointsCommand = rospy.Publisher('/ur5/jointsPosTargetCommand', ManipulatorJoints, queue_size=1)

		# registering to subscribers
		self.sub_joy = rospy.Subscriber('/joy', Joy, self.callback_Joy)

		# kinect_rgb subscriber
		self.sub_kinect_rgb = rospy.Subscriber('/sensor/kinect_rgb', Image, self.callback_kinect_rgb)

		# traction_speed subscriber
		self.sub_traction_speed = rospy.Subscriber('/rosi/command_traction_speed', RosiMovementArray, self.callback_traction_speed)

		# ur5toolCam subscriber
		self.sub_traction_speed = rospy.Subscriber('/sensor/ur5toolCam', Image, self.callback_ur5toolCam)

		# ur5 force torque
		self.sub_traction_speed = rospy.Subscriber('/ur5/forceTorqueSensorOutput', TwistStamped, self.callback_TorqueSensor)

		# gps
		self.sub_traction_speed = rospy.Subscriber('/sensor/gps', NavSatFix, self.callback_gps)

		# defining the eternal loop frequency
		node_sleep_rate = rospy.Rate(10)

		print("Please, press START button to running the model...")

		# eternal loop (until second order)
		while not rospy.is_shutdown():

			arm_command_list = RosiMovementArray()
			traction_command_list = RosiMovementArray()
			arm_joint_list = ManipulatorJoints()

			# mounting the lists
			for i in range(4):

				# ----- treating the traction commands
				traction_command = RosiMovement()

				# mount traction command list
				traction_command.nodeID = i+1

				if self.autoModeStart == True:
					#print("Autonomous mode running...")
					# separates each traction side command
					if i < 2 and self.steering_angle is not None:
						traction_command.joint_var = self.steering_angle[0][i] #self.omega_right  

					if i >= 2 and self.steering_angle is not None:
						traction_command.joint_var = self.steering_angle[0][i] #self.omega_left 

				if self.autoModeStart == False:
					# separates each traction side command
					if i < 2:
						traction_command.joint_var = self.omega_right
					else:
						traction_command.joint_var = self.omega_left

				# appending the command to the list
				traction_command_list.movement_array.append(traction_command)

				# ----- treating the arms commands		
				arm_command = RosiMovement()
		
				# mounting arm command list
				arm_command.nodeID = i+1
				
				# separates each arm side command
				if i == 0 or i == 2:
					arm_command.joint_var = self.arm_front_rotSpeed
				else:
					arm_command.joint_var = self.arm_rear_rotSpeed

				# appending the command to the list
				arm_command_list.movement_array.append(arm_command)

				arm_joint_command = RosiMovement()


			'''		
			Thease are conditions used to move the arm. The callback_Joy change variables states to select and move the joints
			Variables:			
				- self.moveJointLeft and self.moveJointRight => used to rotate joints to the left and right positions.  
			'''
			deltaArm = 2 # angle increment (used to move the arm's joints)

			if self.moveJointLeft == 1 and self.moveJointRight == 0 and self.resetArm == 0:
				print("Theta:", self.thetaJoint)
				self.thetaJoint = self.thetaJoint - deltaArm
			if self.moveJointRight == 1 and self.moveJointLeft == 0 and self.resetArm == 0:
				print("Theta:", self.thetaJoint)
				self.thetaJoint = self.thetaJoint + deltaArm
				self.thetaJoint = self.thetaJoint + deltaArm
			if self.moveJointLeft == 1 and self.moveJointRight == 1 and self.resetArm == 0:
				print("Nothing Done!")
			if self.selectFunction == 1:
				self.thetaJoint = 0
				print("Select Function")
				self.jointSelect = self.jointSelect + 1
				if self.jointSelect >= 6:
					self.jointSelect = 0
				print("Joint Number", self.jointSelect)

			if self.resetArm == 1:
				print("Reset Arm Position")
				self.moveArm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
				self.thetaAll = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # in deg
				self.jointSelect = 0
				self.thetaJoint = 0
				arm_joint_list.joint_variable = self.move_arm_all(self.thetaAll)

			if self.thetaJoint >=180:
				self.thetaJoint = 180
			if self.thetaJoint <=-180:
				self.thetaJoint = -180
			
			if self.resetArm == 0:
				arm_joint_list.joint_variable = self.move_arm_joint(self.thetaJoint, self.jointSelect)
			
			# Publishing topics to vrep simulator
			self.pub_arm.publish(arm_command_list)		
			self.pub_traction.publish(traction_command_list)
			self.pub_kinect_joint.publish()  # -45 < theta < 45 (graus)
			self.pub_jointsCommand.publish(arm_joint_list)  
			
			# sleeps for a while
			node_sleep_rate.sleep()

		# infinite loop
		#while not rospy.is_shutdown():
			# pass

		# enter in rospy spin
		#rospy.spin()


	# Starting usefull functions

	def fire_detection(self, img):
		'''
		This routine is used to detect fire in the vrep simulator. For this purpose,
		it was used a simple color detections function.
		Reference: https://opencv.org/
		'''

		# 1. Define color mask (yellow detection)
		light_color = (39, 255, 255)
		dark_color = (23, 100, 100)

		# 2. Converting BGR to HSV color space
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# 3. Applying the color mask
		mask = cv2.inRange(hsv_img, dark_color, light_color)
		result = cv2.bitwise_and(img, img, mask=mask)

		# 4. Using a gaussian function to extract noises
		kernel_size = 5
		result = cv2.GaussianBlur(result,(kernel_size, kernel_size), 0)

		# 5. Find contours
		_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)	
		for contour in contours:
		        cv2.drawContours(img, contour, -1, (0, 0, 255), 2)	
			print("Take care, fire detected!!")
		
			# In case of fire detection, the GPS coordinates are sent to the map
			plt.plot(self.latitude, self.longitude, color='r', marker="P", label="Fire")
			plt.legend(framealpha=1, frameon=True);
		
		# 6. plot functions
		#cv2.imshow("ROSI Color Detection", img)
		#cv2.waitKey(1)

		return None

	def move_arm_joint(self, theta, joint):
		'''
		Take as input an angle and a corresponding joint and convert from deg to rad
		Inputs: 
			- theta (angle in deg)
			- joint (number os corresponding joint of the arm)
		'''
		grausToRad = 0.0174 # 3.14(rad)/180(deg)
		angInRad = theta*grausToRad # From deg to rad
		self.moveArm[joint] = angInRad # Vetor of the arm joints
		return self.moveArm

	def move_arm_all(self, theta):
		'''
		Take as input a vector of angles in deg and convert to rad
		Input: 
			- theta (angle in deg)
		'''
		grausToRad = 0.0174 # 3.14(rad)/180(deg)

		# convert all input vector from deg to rad
		for i in range(len(theta)):
			theta[i] = theta[i]*grausToRad

		return theta
    
	def save_image(self, folder, frame, countImage):
		'''
		This function is used to save images in a folder to build a dataset for training the CNN
		Inputs:
			- folder (name of the destination folder)
			- frame (frame from cams)
			- countImage (a count variable used to enumarate the current picture)
		'''		
		# 1. Get width and height of the images
		height,width = frame.shape[0],frame.shape[1]  
		
		# 2. Create a empty matrix with the same dimension of the images
		rgb = np.empty((height,width,3),np.uint8) 

		# 3. Save the image in a specified path
		path = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script'+str('/')+folder # Replace with your path folder
		file_name = folder+'_'+str(countImage+self.offset)+'.jpg'
		file_to_save = os.path.join(path,file_name)    
		cv2.imwrite(os.path.join(path,file_to_save), rgb)

		return None

	def save_command_csv(self, countImage, imageName1, imageName2):
		'''
		Save motors commands in a csv file. This is used to train the model for the convolutional neural network (CNN)
		Inputs:
			- countImage (a count variable used to enumarate the current picture)	
			- image ()
		'''
		# 1. Write paths to images
		path_to_imageName1 = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/'+imageName1+'/'   # Replace with your path folder
		path_to_imageName2 = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/'+imageName2+'/'   # Replace with your path folder
		path_to_folder = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/robotCommands/'        # Replace with your path folder

		# 2. Create a csv file and save robot traction commands
		with open(path_to_folder+"driving_log.csv", 'a') as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			file_name = path_to_imageName1+imageName1+'_'+str(countImage+self.offset)+'.jpg'
			file_name_depth = path_to_imageName2+imageName2+'_'+str(countImage+self.offset)+'.jpg'
			filewriter.writerow([path_to_imageName1+file_name, path_to_imageName2+file_name_depth, self.tractionCommand[0][0], 								self.tractionCommand[1][0], self.tractionCommand[2][0], 							self.tractionCommand[3][0]])#, self.mediaVector])
		return None

	def save_map_csv(self, latitude, longitude):
		'''
		This function is used to save GPS data for plotting purpose
		Inputs:
			- latitude and longitude data from GPS (vrep simulator)
		'''
		# 1. Path to save data
		path_to_folder = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/map/' # Replace with your path folder

		# 2. Save data
		with open(path_to_folder+"map_log.csv", 'a') as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			filewriter.writerow([latitude, longitude])
		return None


	def preprocess(self, img):
		'''
		Pipeline to process images from cams. This is used to train the model 
		for the convolutional neural network (CNN)	
		Input: 
			- Image (BRG image from cams)	
		'''	
		# 1. Gaussian blur to damping noises
		image = cv2.GaussianBlur(img, (3,3), 0)

		# 2. Resize image to speed up the CNN training
		image = cv2.resize(image, (64,64), interpolation=cv2.INTER_AREA)

		# 3. BGR to YUV 
		proc_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)	# cv2 loads images as BGR

		return proc_img

	# joystick callback function
	def callback_Joy(self, msg):
		#rospy.loginfo("Joy Test")
		# saving joy commands
		axes_lin = msg.axes[1]
		axes_ang = msg.axes[0]
		self.trigger_left = msg.axes[2]
		self.trigger_right = msg.axes[3]
		button_L = msg.buttons[4]
		button_R = msg.buttons[5]
		record = msg.buttons[10]
		self.autoMode = self.autoMode + msg.buttons[9]
		self.moveJointLeft = msg.buttons[6]
		self.moveJointRight = msg.buttons[7]
		self.selectFunction = msg.buttons[14]
		self.resetArm = msg.buttons[15]

		if record == 1:
			self.save_image_flag = True
			print("Recording data...")
		if record == 0:
			self.save_image_flag = False
			#print("Stop recording data!")
		if self.autoMode % 2 == 0:
			print("Autonomous mode on")
			self.autoModeStart = True
			button_L = 1
		else:
			print("Autonomous mode off")
			self.autoModeStart = False

		# Treats axes deadband
		if axes_lin < 0.15 and axes_lin > -0.15:
			axes_lin = 0

		if axes_ang < 0.15 and axes_ang > -0.15:
			axes_ang = 0

		# treats triggers range
		self.trigger_left = ((-1 * self.trigger_left) + 1) / 2
		self.trigger_right = ((-1 * self.trigger_right) + 1) / 2

		# computing desired linear and angular of the robot
		vel_linear_x = self.max_translational_speed * axes_lin
		vel_angular_z = self.max_rotational_speed * axes_ang

		# -- computes traction command - kinematic math

		# b matrix
		b = np.array([[vel_linear_x],[vel_angular_z]])

		# finds the joints control
		x = np.linalg.lstsq(self.kin_matrix_A, b, rcond=-1)[0]

		# query the sides velocities
		self.omega_right = np.deg2rad(x[0][0])
		self.omega_left = np.deg2rad(x[1][0])

		# -- computes arms command
		# front arms
		if button_R == 1:
			self.arm_front_rotSpeed = self.max_arms_rotational_speed * self.trigger_right
		#else:
		#	self.arm_front_rotSpeed = -1 * self.max_arms_rotational_speed * self.trigger_right

		# rear arms
		if button_L == 1:
			self.arm_rear_rotSpeed = -1 * self.max_arms_rotational_speed * self.trigger_left
		#else:
		#	self.arm_rear_rotSpeed = self.max_arms_rotational_speed * self.trigger_left

	def callback_kinect_rgb(self, msg):
		'''
		Kinect callback function
		Input: 
			- Image from vrep simulator
		'''
		self.camera_image = msg

		# 1. Conversion for opencv 
		try:
			cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
		except CvBridgeError as e:
 			print(e)

		# 2. From RGB to BGR, resize for visualization and flipping
		img_out = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
		img_out = cv2.resize(img_out, None, fx=.6, fy=.6)
		img_out_flip = cv2.flip(img_out, 1)
		self.rgbOut = img_out_flip

		# 3. Call save image function. This is for CNN purpose	
		if self.save_image_flag:
			self.save_image('single_rgb_data', self.rgbOut, self.countImageRGB)

		# 4. Uncomment to display the image
		#cv2.imshow("ROSI Cam RGB", img_out)	
		#cv2.waitKey(1)
		return None

	def callback_ur5toolCam(self, msg):
		'''
		Callback functions to get ur5toolCam image data	
		Input:
			- Image from vrep simulator
		'''
		self.camera_image_arm = msg

		# 1. Conversion for opencv 
		try:
			cv_image = self.bridge.imgmsg_to_cv2(self.camera_image_arm, "rgb8")
		except CvBridgeError as e:
 			print(e)

		# 2. From RGB to BGR, resize for visualization and flipping
		img_out = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
		img_out = cv2.resize(img_out, None, fx=.6, fy=.6)
		self.img_out_arm = cv2.flip(img_out, 1)

		# 3. Call fire detection function
		self.fire_detection(self.img_out_arm)

		# 4. Concatenate ur5 and kinect images
		self.put_image_together()

		return None

	def put_image_together(self):
		'''
		This function is used to concatenate images from kinect and ur5toolCam.
		Besides that, this function calls the routine to predict the velocity 
		of the motors and get directions to follow the path.		
		'''
		# 1. Create a single image with ur5 and kinect ones
		self.concatImage = np.concatenate((self.img_out_arm, self.rgbOut), axis=1)

		# 2. Create a array and call the process function
		image_array = np.asarray(self.concatImage)
		img_out_preprocessed = self.preprocess(image_array)

		# 3. Counting routine to wait robot being ready to go
		if self.autoModeStart == True:# and self.contStart < 301:
			self.contStart = self.contStart + 1
			print("Almost there...", self.contStart)

		# 4. Call function to predict the traction commands
		offset_lap = 2300
		if self.contStart >= 0 and self.contStart < 300:
			#print("####1####")
			self.arm_front_rotSpeed = -1.0 * self.max_arms_rotational_speed * 0.5 #self.trigger_right
			self.arm_rear_rotSpeed = -1.0 * self.max_arms_rotational_speed * 0.5 #self.trigger_left			

		if self.contStart >= 300 and self.contStart < offset_lap: 
			#print("####2####")
			self.steering_angle = model1.predict(img_out_preprocessed[None, :, :, :], batch_size=1)
		
		if self.contStart >= offset_lap and self.contStart < 430 + offset_lap:
			#print("####3####")
			self.steering_angle = model2.predict(img_out_preprocessed[None, :, :, :], batch_size=1)
			self.arm_front_rotSpeed = 0 * self.max_arms_rotational_speed
			self.arm_rear_rotSpeed = 0 * self.max_arms_rotational_speed
	
		if self.contStart >= 415 + offset_lap and self.contStart < 430 + offset_lap:
			#print("####3.1####")
			self.steering_angle = model2.predict(img_out_preprocessed[None, :, :, :], batch_size=1)
			self.steering_angle = self.steering_angle * [[1.05, 1.05, 1.0, 1.0]]

		if self.contStart >= 430 + offset_lap and self.contStart < 530 + offset_lap:
			#print("####4####")
			self.steering_angle = [[0.0, 0.0, 0.0, 0.0]]
			self.arm_front_rotSpeed = 1.8 * self.max_arms_rotational_speed

		if self.contStart >= 530 + offset_lap and self.contStart < 550 + offset_lap: 
			#print("####5####")
			self.steering_angle = [[15.0, 15.0, 15.0, 15.0]]
			self.arm_front_rotSpeed = 0 * self.max_arms_rotational_speed

		if self.contStart >= 550 + offset_lap and self.contStart < 600 + offset_lap:
			#print("####6####")
			self.steering_angle = [[14.0, 14.0, 14.0, 14.0]]
			self.arm_front_rotSpeed = -0.8 * self.max_arms_rotational_speed
			self.arm_rear_rotSpeed = -1.0 * self.max_arms_rotational_speed

		if self.contStart >= 600 + offset_lap and self.contStart < 650 + offset_lap: 
			#print("####7####")
			self.steering_angle = [[0.0, 0.0, 0.0, 0.0]]
			self.arm_rear_rotSpeed = 0 * self.max_arms_rotational_speed

		if self.contStart >= 700 + offset_lap and self.contStart < 710 + offset_lap:
			#print("####8####")
			self.arm_rear_rotSpeed = 1.2 * self.max_arms_rotational_speed

		if self.contStart >= 710 + offset_lap and self.contStart < 730 + offset_lap: 
			#print("####9####")
			self.steering_angle = [[0.0, 0.0, 0.0, 0.0]]
			self.arm_rear_rotSpeed = 0 * self.max_arms_rotational_speed

		if self.contStart >= 730 + offset_lap and self.contStart < 770 + offset_lap: 
			#print("####10####")
			self.steering_angle = [[6.0, 6.0, 6.0, 6.0]]

		if self.contStart >= 770 + offset_lap and self.contStart < 790 + offset_lap: 
			#print("####11####")
			self.steering_angle = [[0.0, 0.0, 0.0, 0.0]]

		if self.contStart >= 790 + offset_lap and self.contStart < 890 + offset_lap: 
			#print("####12####")
			self.steering_angle = [[-12.0, -12.0, -12.0, -12.0]]

		if self.contStart >= 890 + offset_lap:
			#print("####13####")
			self.steering_angle = [[0.0, 0.0, 0.0, 0.0]]

		# 5. Call save functions for tranning the CNN
		if self.save_image_flag:
			self.save_image('rgb_data', self.concatImage, self.countImageRGB)
			self.save_command_csv(self.countImageRGB, 'single_rgb_data', 'rgb_data')
			self.countImageRGB = self.countImageRGB+1	

		# 6. Uncomment to display images	
		#cv2.imshow("ROSI Cams", self.concatImage)
		#cv2.waitKey(1)
		return None


	def callback_traction_speed(self, msg):
		'''
		Get traction speed commands from vrep
		Input:
			- Traction commands from vrep simulator
		'''
		self.robotMovement = msg
		movement_array = [[p.joint_var] for p in self.robotMovement.movement_array]
		self.tractionCommand = movement_array
		return None

	def callback_TorqueSensor(self, msg):
		'''
		Get torque sensor data from vrep
		Input:
			- UR-5 Force/Torque sensor output. It gives two vector of linear 
			and angular forces and torques, respectively. Axis order is x, y, z.
		'''
		#print("Torque Sensor Test", msg)
		return None
	
	def callback_gps(self, msg):
		'''
		Get GPS data from vrep
		Input:
			- latitude and longitude data from vrep simulator
		'''
		# 1. Uncomment to save data
		#self.save_map_csv(msg.latitude, msg.longitude)

		# 2. Definning pointer variables
		self.latitude = msg.latitude
		self.longitude = msg.longitude

		# 3. For plotting purpose, we get a sample after self.gpsInc measurements 
		self.gpsInc = self.gpsInc + 1
		if self.gpsInc == 20:
			self.latitudes.append(self.latitude)
			self.longitudes.append(self.longitude)
			plt.title("GPS tracking")
			plt.plot(self.latitudes, self.longitudes, color='b', label='Path')
			plt.xlabel('Latitude')
			plt.ylabel('Longitude')
			plt.legend(framealpha=1, frameon=True);
			plt.grid(True)
			plt.pause(0.001)
			self.gpsInc = 0
			plt.savefig('/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/map/map.png')
		return None

	# ---- Support Methods --------

	# -- Method for compute the skid-steer A kinematic matrix
	@staticmethod
	def compute_kinematicAMatrix(var_lambda, wheel_radius, ycir):

		# kinematic A matrix 
		matrix_A = np.array([[var_lambda*wheel_radius/2, var_lambda*wheel_radius/2],
							[(var_lambda*wheel_radius)/(2*ycir), -(var_lambda*wheel_radius)/(2*ycir)]])

		return matrix_A

# instaciate the node
if __name__ == '__main__':

	# initialize the node
	rospy.init_node('rosi_example_node', anonymous=True)

	# instantiate the class
	try:
		node_obj = RosiNodeClass()
	except rospy.ROSInterruptException: pass
