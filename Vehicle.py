import numpy as np
from collections import deque
import logging

class Vehicle():
	
	def __init__(self):
		self.reset()		
	def reset(self):
		self.len_queue = 15
		self.detected = False
		self.n_detections = 0
		self.n_non_detections = 0
		self.ack_threshold = 20
		self.non_detect_threshold = 1
		self.x_pixels = deque(maxlen = self.len_queue)
		self.y_pixels = deque(maxlen = self.len_queue)
		self.recent_x_left = deque(maxlen = self.len_queue)
		self.recent_x_right = deque(maxlen = self.len_queue)
		self.recent_y_top = deque(maxlen = self.len_queue)
		self.recent_y_bottom = deque(maxlen = self.len_queue)
		self.current_x_left = None
		self.current_x_right = None
		self.current_y_top = None
		self.current_y_bottom = None
		self.margin = 10

	def verify_previously_seen(self,labeled_bboxes,imshape):
		detected = False
		detected_labels = []
		for labeled_bbox in labeled_bboxes:
			bbox = labeled_bbox[1]
			label  = labeled_bbox[0]

			x_left = bbox[0][0]
			y_top = bbox[0][1]
			x_right = bbox[1][0]
			y_bottom = bbox[1][1]

			margin_x_left = max(self.current_x_left-self.margin,0)
			margin_x_right = min(self.current_x_right+self.margin,imshape[1])
			margin_y_top = max(self.current_y_top-self.margin,0)
			margin_y_bottom = min(self.current_y_bottom+self.margin,imshape[0])

			r1 = (margin_x_left,margin_y_top,margin_x_right,margin_y_bottom)
			r2 = (x_left,y_top,x_right,y_bottom)

			if(self.determine_overlap(r1,r2)):
				
				#we've detected an intersection
				detected = True
				self.n_detections+=1
			
				selfwidth = self.current_x_right - self.current_x_left
				selfheight = self.current_y_bottom - self.current_y_top

				boxwidth = x_right - x_left
				boxheight = y_bottom - y_top
				detected_labels.append(labeled_bbox)

				diff_width = selfwidth-boxwidth
				diff_height = selfheight-boxheight

				if(np.absolute(diff_width)>(0.25*selfwidth)):
					#adjust x values
					if (diff_width<0):  
						#when the box is bigger
						x_left +=  (0.25*selfwidth)/2
						x_right -= (0.25*selfwidth)/2
					else :		  
						x_left -=  (0.25*selfwidth)/2
						x_right += (0.25*selfwidth)/2
				
				if(np.absolute(diff_height)>(0.25*selfheight)):
					#adjust x values
					if (diff_height<0):  
						#when the box is bigger
						y_top +=  (0.25*selfheight)/2
						y_bottom -= (0.25*selfheight)/2
					else:		  
						y_top -=  (0.25*selfheight)/2
						y_bottom += (0.25*selfheight)/2

				#add to list of positions
				self.recent_x_left.append(x_left)
				self.recent_x_right.append(x_right)
				self.recent_y_top.append(y_top)
				self.recent_y_bottom.append(y_bottom)

				self.current_x_left = np.int(np.average(self.recent_x_left))
				self.current_x_right = np.int(np.average(self.recent_x_right))
				self.current_y_top = np.int(np.average(self.recent_y_top))
				self.current_y_bottom = np.int(np.average(self.recent_y_bottom))

		if(detected==False):
			self.n_non_detections+=1
		
		return set(detected_labels)
	
	def is_drawable(self):
		if self.n_detections> self.ack_threshold:
			return True
		else:
			return False

	def is_removable(self):
		if self.n_non_detections>=self.non_detect_threshold:
			if(self.n_detections>20):
				self.n_detections -=5
				return False
			else:
				return True
		else:
			return False

	def determine_overlap(self,rect1,rect2):
		if(rect1[0]>=rect2[2] or rect2[0]>=rect1[2]):
			return False
		if(rect1[3]<=rect2[1] or rect2[3]<=rect1[1]):
			return False
		return True
