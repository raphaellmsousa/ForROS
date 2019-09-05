#!/usr/bin/env python2.7
import rospy
import numpy as np
from rosi_defy.msg import RosiMovement
from rosi_defy.msg import RosiMovementArray
from rosi_defy.msg import ManipulatorJoints
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from sensor_msgs.msg import PointCloud
from rosi_defy.msg import HokuyoReading

import os
import os.path

import csv

from keras.models import load_model
##################################################################################
#                          Uncomment do use the model
##################################################################################
model = load_model('/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/model.h5')
model._make_predict_function()

##################################################################################
#                          Instructons
##################################################################################
# Press button 10 (depends on your control configuration) to record data
# Press button 9 keep pressing to running the model 
# Tensorflow version 1.14.0
# Keras version 2.2.5

class RosiNodeClass():

	# class attributes
	max_translational_speed = 5*6 # in [m/s]
	max_rotational_speed = 10*7 # in [rad/s]
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
		self.autoModeStart = False
		self.countHokuyo = 0
		self.mediaVector = 0
		
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

			for i in range(6):
				# separates each arm side command
				if i == 0:
					arm_joint_command = 3.14 #180 graus
				else:
					arm_joint_command = 0

				# appending the command to the list for the arm
				arm_joint_list.joint_variable.append(arm_joint_command)

			# publishing
			self.pub_arm.publish(arm_command_list)		
			self.pub_traction.publish(traction_command_list)
			self.pub_kinect_joint.publish()  # -45 < theta < 45 (graus)
			self.pub_jointsCommand.publish(arm_joint_list)  
			#print(arm_joint_list)
			
			# sleeps for a while
			node_sleep_rate.sleep()

		# infinite loop
		#while not rospy.is_shutdown():
			# pass

		# enter in rospy spin
		#rospy.spin()

	# Here is your draw_boxes function from the previous exercise
	def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=1):
	    # Make a copy of the image
	    imcopy = np.copy(img)
	    # Iterate through the bounding boxes
	    for bbox in bboxes:
	        # Draw a rectangle given bbox coordinates
        	cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	    # Return the image copy with boxes drawn
	    return imcopy
    
    
	# Define a function to search for template matches
	# and return a list of bounding boxes
	def find_matches(self, img, template_list):
    	# Define an empty list to take bbox coords
	    bbox_list = []
	    # Define matching method
	    # Other options include: cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR',
	    #         'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
	    method = cv2.TM_CCOEFF_NORMED
	    # Iterate through template list
	    for temp in template_list:
        	# Read in templates one by one
        	tmp = cv2.imread(temp)
        	# Use cv2.matchTemplate() to search the image
        	result = cv2.matchTemplate(img, tmp, method)
		threshold = 0.9
		loc = np.where( result >= threshold) 
		if loc:
        		# Use cv2.minMaxLoc() to extract the location of the best match
        		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
		else:
			pass
        	# Determine a bounding box for the match
        	w, h = (tmp.shape[1], tmp.shape[0])
        	if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        	    top_left = min_loc
        	else:
        	    top_left = max_loc
        	bottom_right = (top_left[0] + w, top_left[1] + h)
        	# Append bbox position to list
        	bbox_list.append((top_left, bottom_right))
        	# Return the list of bounding boxes
        
	    return bbox_list

	# Save image to a folder
	def save_image(self, folder, frame, countImage):
		height,width = frame.shape[0],frame.shape[1] #get width and height of the images 
		rgb = np.empty((height,width,3),np.uint8) 
		path = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script'+str('/')+folder # Replace with your path folder
		file_name = folder+'_'+str(countImage)+'.jpg'
		file_to_save = os.path.join(path,file_name)    
		cv2.imwrite(os.path.join(path,file_to_save), rgb)
		return None

	def save_command_csv(self, count, image, image_depth):
		path_to_image = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/'+image+'/'
		path_to_image_depth = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/'+image_depth+'/' # Replace with your path folder
		path_to_folder = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/robotCommands/'        # Replace with your path folder
		with open(path_to_folder+"driving_log.csv", 'a') as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			file_name = path_to_image+image+'_'+str(count)+'.jpg'
			file_name_depth = path_to_image_depth+image_depth+'_'+str(count)+'.jpg'
			filewriter.writerow([path_to_image+file_name, path_to_image_depth+file_name_depth, self.tractionCommand[0][0], 								self.tractionCommand[1][0], self.tractionCommand[2][0], 							self.tractionCommand[3][0]])#, self.mediaVector])
		return None

	def preprocess(self, img):
		image = cv2.GaussianBlur(img, (3,3), 0)
		image = cv2.resize(image, (64,64), interpolation=cv2.INTER_AREA)
		proc_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)	# cv2 loads images as BGR
		return proc_img

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
		autoMode = 1#msg.buttons[9]

		if record == 1:
			self.save_image_flag = True
			print("Recording data...")
		if record == 0:
			self.save_image_flag = False
			#print("Stop recording data!")
		if autoMode == 1:
			self.autoModeStart = True
		if autoMode == 0:
			self.autoModeStart = False

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
		img_out = cv2.resize(img_out, None, fx=.6, fy=.6)
		img_out = cv2.flip(img_out, 1)
		#templist = ['/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/fire1.jpg']
		#bboxes = self.find_matches(img_out, templist)
		#print(bboxes)
		#img_out = self.draw_boxes(img_out, bboxes)
		#gray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
		cv2.imshow("ROSI Cam rgb", img_out)
		#cv2.imshow("ROSI Cam gray", gray)
		image_array = np.asarray(img_out)
		img_out_preprocessed = self.preprocess(image_array)
		if self.autoModeStart == True:
			self.steering_angle = model.predict(img_out_preprocessed[None, :, :, :], batch_size=1)
			print(self.steering_angle)
		if self.save_image_flag:
			self.countImageRGB = self.countImageRGB+1
			self.save_image('rgb_data', img_out, self.countImageRGB)
			self.save_command_csv(self.countImageRGB, 'rgb_data', 'depth_data')
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
		vectorSize = len(self.hokuyoOut.reading)
		sumVector = 0
		for i in range(vectorSize):
			sumVector = sumVector + abs(self.hokuyoOut.reading[i])
		self.mediaVector = sumVector/vectorSize
		#print(self.hokuyoOut)
		print(self.mediaVector)	
		#print(self.hokuyoOut.reading[self.countHokuyo]) 
		#print(self.countHokuyo)
		#self.countHokuyo = self.countHokuyo + 1
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

