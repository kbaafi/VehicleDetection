import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from FeatureExtractor import *
from Classifier import *
from scipy.ndimage.measurements import label
from feature_utils import *
from imsearch_utils import *
from Vehicle import *
import logging
from calibration import *


class VehicleDetectionPipeline():
	def __init__(self):
		self.reset()

	def reset(self):
		self.vehicle_list = []
		self.classifier = None
		with open('clf.p','rb') as f:
			self.classifier = pickle.load(f)
		self.ret,self.dist_coeffs,self.cam_matrix,self.rvecs,self.tvecs = calibrate_camera('camera_cal','calibration*.jpg',(9,6),(720,1280))

	def process(self,original_img):
		ret_img = np.array(undistort(original_img,self.cam_matrix,self.dist_coeffs))
		ycrcb_img = np.copy(ret_img).astype(np.float32)/255
		ycrcb_img = cv2.cvtColor(ycrcb_img,cv2.COLOR_RGB2YCrCb)

		search_area = get_search_area_settings(); 
		cropped_img = ycrcb_img[search_area[0]:,search_area[1]:,:]

		img_shape = np.shape(original_img)
		bboxes = slide_window_multiscalar(img_shape)

		windows_list = search_classify_img(bboxes,cropped_img,self.classifier,hist_bins = 32)

		r_img = add_heat(cropped_img,windows_list)

		th_img = apply_threshold(r_img,2)

		
		labels = label(th_img)
		labeled_bboxes = set(get_labeled_bboxes(labels,search_area))

		ids = set(range(1,labels[1]+1))

		detected_boxes = []
		
		
		if(len(self.vehicle_list)>0):
			for vehicle in self.vehicle_list:
				detected_boxes_v = vehicle.verify_previously_seen(labeled_bboxes,img_shape)
				labeled_bboxes =  labeled_bboxes - detected_boxes_v
			for labeled_bbox in labeled_bboxes:
				self.create_vehicle(labeled_bbox)			 
		else:
			#create a the vehicle object
			for labeled_bbox in labeled_bboxes:
				self.create_vehicle(labeled_bbox)
		
		for vehicle in self.vehicle_list:
			if vehicle.is_removable():
				self.vehicle_list.remove(vehicle)
				
		return self.draw_vehicles(np.copy(ret_img), self.vehicle_list)

	def select_labeled_bbox(self,labeled_bboxes,idx):
		for element in labeled_bboxes:
			if(idx==element[0]):
				return element
	def create_vehicle(self,labeled_bbox):
		vehicle = Vehicle()
		vehicle.n_detections  = 1
		vehicle.n_non_detections = 0
		box_corners = labeled_bbox[1]
		vehicle.recent_x_left.append(box_corners[0][0])
		vehicle.recent_x_right.append(box_corners[1][0])
		vehicle.recent_y_top.append(box_corners[0][1])
		vehicle.recent_y_bottom.append(box_corners[1][1])

		vehicle.current_x_left = box_corners[0][0]
		vehicle.current_x_right = box_corners[1][0]
		vehicle.current_y_top = box_corners[0][1]
		vehicle.current_y_bottom = box_corners[1][1]
		self.vehicle_list.append(vehicle)
	
	def draw_vehicles(self,img, vehicle_list):
		for vehicle in vehicle_list:
			if vehicle.is_drawable():
				bbox = ((vehicle.current_x_left,vehicle.current_y_top),(vehicle.current_x_right,vehicle.current_y_bottom))
				cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 4)
		return img
		
		
	
