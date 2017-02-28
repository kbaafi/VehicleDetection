import numpy as np
import cv2
from feature_utils import *
from functools import partial
from multiprocessing import Pool, freeze_support
import copy


def add_heat(img,bbox_list):
	c_img = np.zeros_like(img[:,:,0])
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		#c_img[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
		c_img[box[0][0]:box[1][0], box[0][1]:box[1][1]] += 1

	# Return updated heatmap
	return c_img

def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap

def get_labeled_bboxes(labels,search_area_settings):
	labeled_bboxes = []
	for car_number in range(1,labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox)+search_area_settings[1], np.min(nonzeroy)+search_area_settings[0]), 
			(np.max(nonzerox)+search_area_settings[1], np.max(nonzeroy)+search_area_settings[0]))
		labeled_bboxes.append((car_number,bbox))

	return labeled_bboxes

def draw_labeled_bboxes(img, labels,search_area_settings):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox)+search_area_settings[1], np.min(nonzeroy)+search_area_settings[0]), 
			(np.max(nonzerox)+search_area_settings[1], np.max(nonzeroy)+search_area_settings[0]))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 4)
		# Return the image
	return img


def search_classify_img(window_list,y_img,clf,spatial_size=(32, 32),hist_bins = 32):
	hot_windows = []
	for window in window_list:
		startx = window[0][0]
		starty = window[0][1]
		endx = window[1][0]
		endy = window[1][1]

		col_img = cv2.resize(y_img[startx:endx,starty:endy],(64,64))
		hog_feats = get_hog_features_all(col_img, 9, 8, 2, 
                vis=False, feature_vec=True)
		hog_feats = np.ravel(hog_feats)
    
		bin_spatial_feats = bin_spatial(col_img, size=spatial_size)
		col_hist_feats = color_hist(col_img, nbins=hist_bins)

		img_features = [];
		img_features.extend(bin_spatial_feats)
		img_features.extend(col_hist_feats)
		img_features.extend(hog_feats)
		score = clf.predict(img_features)
		if(score==1):
			hot_windows.append(window)
	return hot_windows

def pooled_img_search(window,img,clf):
	spatial_size=(32, 32)
	hist_bins = 32

	startx = window[0][0]
	starty = window[0][1]
	endx = window[1][0]
	endy = window[1][1]

	col_img = cv2.resize(img[startx:endx,starty:endy],(64,64))
	hog_feats = get_hog_features_all(col_img, 9, 8, 2, 
        vis=False, feature_vec=True)
	hog_feats = np.ravel(hog_feats)

	bin_spatial_feats = bin_spatial(col_img, size=spatial_size)
	col_hist_feats = color_hist(col_img, nbins=hist_bins)

	img_features = [];
	img_features.extend(bin_spatial_feats)
	img_features.extend(col_hist_feats)
	img_features.extend(hog_feats)
	score = clf.predict(img_features)

	return (score,window)

def search_classify_img_pooled(window_list,y_img,classifier):
	freeze_support()
	pool = Pool(4)
	scored_windows = pool.map(partial(pooled_img_search,img = y_img,clf = classifier),window_list)
	hot_windows = []
	for element in scored_windows:
		if element[0]==1:
			hot_windows.append(element[1])
	return hot_windows
def draw_boxes(img, bboxes, color=(0, 255, 0), thick=4):
	# Make a copy of the image
	imcopy = np.copy(img)
	# Iterate through the bounding boxes
	for bbox in bboxes:
	# Draw a rectangle given bbox coordinates
		cv2.rectangle(imcopy, (bbox[0][1],bbox[0][0]), (bbox[1][1],bbox[1][0]), color, thick)
	# Return the image copy with boxes drawn
	return imcopy
