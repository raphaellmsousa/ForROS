#!/usr/bin/env python2.7
# ROSI Challenge
# 09/2019
# Team: ForROS
import rospy
import numpy as np
from rosi_defy.msg import RosiMovement
from rosi_defy.msg import RosiMovementArray
from rosi_defy.msg import ManipulatorJoints
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistStamped


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
#                          Uncomment to use the model
##################################################################################
model = load_model('/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/model.h5')
model._make_predict_function()

##################################################################################
#                          Instructions
##################################################################################
# Press button 10 (depends on your control configuration) to record data
# Press button 9 keep pressing to running the model 
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
		self.autoModeStart = False
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
	
			deltaArm = 2

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

			# publishing
			self.pub_arm.publish(arm_command_list)		
			self.pub_traction.publish(traction_command_list)
			#self.pub_kinect_joint.publish(.1)  # -45 < theta < 45 (graus)
			self.pub_jointsCommand.publish(arm_joint_list)  
			#print(arm_joint_list)
			
			# sleeps for a while
			node_sleep_rate.sleep()

		# infinite loop
		#while not rospy.is_shutdown():
			# pass

		# enter in rospy spin
		#rospy.spin()

	def fire_detection(self, img):
		light_color = (39, 255, 255)
		dark_color = (23, 100, 100)
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv_img, dark_color, light_color)
		result = cv2.bitwise_and(img, img, mask=mask)
		kernel_size = 5
		result = cv2.GaussianBlur(result,(kernel_size, kernel_size), 0)
		_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)	
		for contour in contours:
		        cv2.drawContours(img, contour, -1, (0, 0, 255), 2)	
			print("Take care, fire detected!!")
		#cv2.imshow("ROSI Color Detection", img)
		#cv2.waitKey(1)
		return None

	def roll_detection(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		low_threshold = 100 #type a positive value
		high_threshold = 150 #type a positive value
		edges = cv2.Canny(gray, low_threshold, high_threshold)

		#cv2.imshow("ROSI Roll Detection", edges)
		#cv2.waitKey(1)
		return None

	def move_arm_joint(self, theta, joint):
		grausToRad = 0.0174
		angInRad = theta*grausToRad
		self.moveArm[joint] = angInRad
		return self.moveArm

	def move_arm_all(self, theta):
		grausToRad = 0.0174
		for i in range(len(theta)):
			theta[i] = theta[i]*grausToRad
		return theta
    
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
		color_select = np.copy(img)
		color_select[:,:,0] = 0
		color_select[:,:,1] = 0 
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
		#color_select_HLS_S = img[:,:,2]		
		img[:,:,0] = 0
		img[:,:,1] = 0 
		final = img * (color_select)
		#cv2.imshow("test", final)
        	# Use cv2.matchTemplate() to search the image
        	result = cv2.matchTemplate(final, tmp, method)
		threshold = 0.19
       		# Use cv2.minMaxLoc() to extract the location of the best match
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)	
		#print(min_val)	
		if abs(min_val) >= threshold:
			print("Roll Founded")
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
		if abs(min_val) <= threshold:
			pass
        
	    return bbox_list

	# Here is your draw_boxes function from the previous exercise
	def draw_boxes(self, img, bboxes, color=(0, 255, 0), thick=2):
	    # Make a copy of the image
	    imcopy = np.copy(img)
	    # Iterate through the bounding boxes
	    for bbox in bboxes:
	        # Draw a rectangle given bbox coordinates
        	cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	    # Return the image copy with boxes drawn
	    return imcopy

	# Save image to a folder
	def save_image(self, folder, frame, countImage):
		height,width = frame.shape[0],frame.shape[1] #get width and height of the images 
		rgb = np.empty((height,width,3),np.uint8) 
		path = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script'+str('/')+folder # Replace with your path folder
		file_name = folder+'_'+str(countImage+self.offset)+'.jpg'
		file_to_save = os.path.join(path,file_name)    
		cv2.imwrite(os.path.join(path,file_to_save), rgb)
		return None

	def save_command_csv(self, count, image, image_depth):
		path_to_image = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/'+image+'/'
		path_to_image_depth = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/'+image_depth+'/' # Replace with your path folder
		path_to_folder = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/robotCommands/'        # Replace with your path folder
		with open(path_to_folder+"driving_log.csv", 'a') as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			file_name = path_to_image+image+'_'+str(count+self.offset)+'.jpg'
			file_name_depth = path_to_image_depth+image_depth+'_'+str(count+self.offset)+'.jpg'
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
		button_L = 1 #msg.buttons[4]
		button_R = msg.buttons[5]
		record = msg.buttons[10]
		autoMode = msg.buttons[9]
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
		self.camera_image = msg
		try:
			cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
		except CvBridgeError as e:
 			print(e)
		img_out = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
		img_out = cv2.resize(img_out, None, fx=.6, fy=.6)
		img_out_flip = cv2.flip(img_out, 1)
		self.rgbOut = img_out_flip	
		if self.save_image_flag:
			self.save_image('single_rgb_data', self.rgbOut, self.countImageRGB)
		cv2.imshow("ROSI Cam RGB", img_out)	
		cv2.waitKey(1)
		return None

	def callback_ur5toolCam(self, msg):
		self.camera_image_arm = msg
		try:
			cv_image = self.bridge.imgmsg_to_cv2(self.camera_image_arm, "rgb8")
		except CvBridgeError as e:
 			print(e)
		img_out = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
		img_out = cv2.resize(img_out, None, fx=.6, fy=.6)
		self.img_out_arm = cv2.flip(img_out, 1)
		self.fire_detection(self.img_out_arm)
		#cv2.imshow("ROSI ur5toolCam", self.img_out_arm)
		#self.roll_detection(self.img_out_arm)
		templist = ['/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/roll.jpg']
		bboxes = self.find_matches(self.img_out_arm, templist)
		img_out_crop = self.draw_boxes(self.img_out_arm, bboxes)
		self.put_image_together()
		return None

	def put_image_together(self):
		self.concatImage = np.concatenate((self.img_out_arm, self.rgbOut), axis=1)
		image_array = np.asarray(self.concatImage)
		img_out_preprocessed = self.preprocess(image_array)
		if self.autoModeStart == True and self.contStart < 301:
			self.contStart = self.contStart + 1
			print(self.contStart)
		if self.contStart >= 300: 
			self.steering_angle = model.predict(img_out_preprocessed[None, :, :, :], batch_size=1)	
		if self.save_image_flag:
			self.save_image('rgb_data', self.concatImage, self.countImageRGB)
			self.save_command_csv(self.countImageRGB, 'single_rgb_data', 'rgb_data')
			self.countImageRGB = self.countImageRGB+1		
		cv2.imshow("ROSI Cams", self.concatImage)
		cv2.waitKey(1)
		return None

	def callback_traction_speed(self, msg):
		self.robotMovement = msg
		movement_array = [[p.joint_var] for p in self.robotMovement.movement_array]
		self.tractionCommand = movement_array
		return None

	def callback_TorqueSensor(self, msg):
		print("Torque Sensor Test", msg)
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
