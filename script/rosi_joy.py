#!/usr/bin/env python2
import rospy
import numpy as np
from rosi_defy.msg import RosiMovement
from rosi_defy.msg import RosiMovementArray
from rosi_defy.msg import ManipulatorJoints
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from sensor_msgs.msg import PointCloud
from rosi_defy.msg import HokuyoReading

import os
import os.path

current_directory = os.getcwd()

class RosiNodeClass():

	# class attributes
	max_translational_speed = 5*5 # in [m/s]
	max_rotational_speed = 10*5 # in [rad/s]
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

		# computing the kinematic A matrix
		self.kin_matrix_A = self.compute_kinematicAMatrix(self.var_lambda, self.wheel_radius, self.ycir)

		# sends a message to the user
		rospy.loginfo('Rosi_joy node started')

		# registering to publishers
		self.pub_traction = rospy.Publisher('/rosi/command_traction_speed', RosiMovementArray, queue_size=1)
		self.pub_arm = rospy.Publisher('/rosi/command_arms_speed', RosiMovementArray, queue_size=1)

		# registering to subscribers
		self.sub_joy = rospy.Subscriber('/joy', Joy, self.callback_Joy)

		# kinect_rgb subscriber
		self.sub_kinect_rgb = rospy.Subscriber('/sensor/kinect_rgb', Image, self.callback_kinect_rgb)

		# kinect_depth subscriber
		self.sub_kinect_depth = rospy.Subscriber('/sensor/kinect_depth', Image, self.callback_kinect_depth)

		# velodyne subscriber
		self.sub_velodyne = rospy.Subscriber('/sensor/velodyne', PointCloud, self.callback_velodyne)

		# hokuyo subscriber
		self.sub_hokuyo = rospy.Subscriber('/sensor/hokuyo', HokuyoReading, self.callback_hokuyo)

		# traction_speed subscriber
		self.sub_traction_speed = rospy.Subscriber('/rosi/command_traction_speed', RosiMovementArray, self.callback_traction_speed)

		# defining the eternal loop frequency
		node_sleep_rate = rospy.Rate(10)

		# eternal loop (until second order)
		while not rospy.is_shutdown():

			arm_command_list = RosiMovementArray()
			traction_command_list = RosiMovementArray()

			# mounting the lists
			for i in range(4):

				# ----- treating the traction commands
				traction_command = RosiMovement()

				# mount traction command list
				traction_command.nodeID = i+1

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

			# publishing
			self.pub_arm.publish(arm_command_list)		
			self.pub_traction.publish(traction_command_list)

			# sleeps for a while
			node_sleep_rate.sleep()

		# infinite loop
		#while not rospy.is_shutdown():
			# pass

		# enter in rospy spin
		#rospy.spin()

	# Save image to a folder
	def save_image(self, folder, frame, countImage):
		height,width = frame.shape[0],frame.shape[1] #get width and height of the images 
		rgb = np.empty((height,width,3),np.uint8) 
		path = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script'+str('/')+folder # Replace with your path folder
		print(path)
		file_name = folder+'_'+str(countImage)+'.jpg'
		file_to_save = os.path.join(path,file_name)    
		cv2.imwrite(os.path.join(path,file_to_save), rgb)
		return None

	def save_command(self, count, data, saveToFile):
		flagW = "w"
		flagA = "a"		
		file1 = open("/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/robotCommands/"+saveToFile, self.flag) # Replace with your path folder
		file1.write(str(self.countImageRGB) + "," + str(data)+"\n") 
		if self.flag == flagW:
			self.flag = flagA
		file1.close()
		return None

	# joystick callback function
	def callback_Joy(self, msg):
		#rospy.loginfo("Joy Test")
		# saving joy commands
		axes_lin = msg.axes[1]
		axes_ang = msg.axes[0]
		trigger_left = msg.axes[2]
		trigger_right = msg.axes[3]
		button_L = msg.buttons[4]
		button_R = msg.buttons[5]
		record = msg.buttons[10]

		if record == 1:
			self.save_image_flag = True
		if record == 0:
			self.save_image_flag = False

		# Treats axes deadband
		if axes_lin < 0.15 and axes_lin > -0.15:
			axes_lin = 0

		if axes_ang < 0.15 and axes_ang > -0.15:
			axes_ang = 0

		# treats triggers range
		trigger_left = ((-1 * trigger_left) + 1) / 2
		trigger_right = ((-1 * trigger_right) + 1) / 2

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
			self.arm_front_rotSpeed = self.max_arms_rotational_speed * trigger_right
		else:
			self.arm_front_rotSpeed = -1 * self.max_arms_rotational_speed * trigger_right

		# rear arms
		if button_L == 1:
			self.arm_rear_rotSpeed = -1 * self.max_arms_rotational_speed * trigger_left
		else:
			self.arm_rear_rotSpeed = self.max_arms_rotational_speed * trigger_left
	
	# kinect callback function
	def callback_kinect_rgb(self, msg):
		#rospy.loginfo("Test Image Callback")
		self.camera_image = msg
		try:
			cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
		except CvBridgeError as e:
 			print(e)
		img_out = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
		img_out = cv2.resize(img_out, None, fx=.5, fy=.5)
		img_out = cv2.flip(img_out, 1)
		#gray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
		#cv2.imshow("ROSI Cam rgb", img_out)
		#cv2.imshow("ROSI Cam gray", gray)
		if self.save_image_flag:
			self.countImageRGB = self.countImageRGB+1
			self.save_image('rgb_data', img_out, self.countImageRGB)
			self.save_command(self.countImageRGB, self.tractionCommand, "commands")
			#self.save_command(self.countImageRGB, self.velodyne, "velodyne/velodyne")
		cv2.waitKey(1)
		return None

	# kinect callback function
	def callback_kinect_depth(self, msg):
		#rospy.loginfo("Test Image Callback")
		self.camera_image_depth = msg
		try:
			cv_image_depth = self.bridge.imgmsg_to_cv2(self.camera_image_depth, "rgb8")
		except CvBridgeError as e:
 			print(e)
		img_out_depth = cv2.cvtColor(cv_image_depth, cv2.COLOR_RGB2BGR)
		img_out_depth = cv2.resize(img_out_depth, None, fx=.5, fy=.5)
		img_out_depth = cv2.flip(img_out_depth, 1)
		#gray_depth = cv2.cvtColor(img_out_depth, cv2.COLOR_BGR2GRAY)
		#cv2.imshow("ROSI Cam depth", img_out_depth)
		#cv2.imshow("ROSI Cam gray_depth", gray_depth)
		if self.save_image_flag:
			self.countImageDepth = self.countImageDepth+1
			self.save_image('depth_data', img_out_depth, self.countImageDepth)
		cv2.waitKey(1)
		return None

	# velodyne callback function
	def callback_velodyne(self, msg):
		#rospy.loginfo("Test Velodyne Callback")
		self.velodyneOut = msg
		points = [[p.x, p.y, p.z, 1] for p in self.velodyneOut.points]
		self.velodyne = points
		#print(self.velodyneOut)
		return None

	# hokuyo callback function
	#https://www.youtube.com/watch?v=RFNNsDI2b6c
	def callback_hokuyo(self, msg):
		#rospy.loginfo("Test Hokuyo Callback")
		self.hokuyoOut = msg
		#print(self.hokuyoOut.reading[0]) # We have 135 points: from 0 to 134
		return None
	
	def callback_traction_speed(self, msg):
		#rospy.loginfo("Test traction_speed Callback")
		self.robotMovement = msg
		movement_array = [[p.joint_var] for p in self.robotMovement.movement_array]
		self.tractionCommand = movement_array
		#print(self.tractionCommand)		
		#print(self.robotMovement.movement_array[0])
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

