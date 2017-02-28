import numpy as np
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle
from feature_utils import *
from sklearn.preprocessing import StandardScaler
import cv2

class Classifier():
	def __init__(self):
		self.X_train = []
		self.X_test = []
		self.y_train = []
		self.y_test= []
		self.classifier = None
		self.scaler = None

	def  load_data_from_feature_extractor(self,fe):
		self.X_train = fe.training_features
		self.X_test = fe.test_features
		self.y_train = fe.training_labels
		self.y_test = fe.test_features

	def load_data(self,training_features,test_features,training_labels,test_labels):
		self.X_train = training_features
		self.X_test = test_features
		self.y_train = training_labels
		self.y_test = test_features

	def load_data_from_pickle(self,filename):
		with open(filename,'rb') as f:
			self.X_train = pickle.load(f)
			self.y_train = pickle.load(f)
			self.X_test = pickle.load(f)
			self.y_test = pickle.load(f)

	def train_SVM(self,filename = None):
		self.classifier = LinearSVC()
		
		self.scaler = StandardScaler().fit(self.X_train)
		
		self.X_train  = self.scaler.transform(self.X_train)
		self.X_test = self.scaler.transform(self.X_test)

		self.classifier.fit(self.X_train,self.y_train)

		score = round(self.classifier.score(self.X_test, self.y_test), 4)

		if(filename is not None):
			joblib.dump(self.classifier, filename+'.pkl') 
		
		# After training clear data
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		return score

	def load_classifier_model(self,filename):
		self.classifier = joblib.load(filename)

	def predict(self,y):
		scaled_y = self.scaler.transform(np.array(y).reshape(1, -1))
		prediction = self.classifier.predict(scaled_y)
		return prediction
"""
	def search_classify_img(self,window_list,y_img,y_hog,spatial_size=(32, 32), hist_bins=32):
		hot_windows = []
		for window in window_list:
			startx = window[0][0]
			starty = window[0][1]
			endx = window[1][0]
			endy = window[1][1]
		
			hog_feats = cv2.resize(y_hog[startx:endx,starty:endy],(64,64))
			col_img = cv2.resize(y_img[startx:endx,starty:endy],(64,64))

			bin_spatial_feats = bin_spatial(col_img, size=spatial_size)
			col_hist_feats = color_hist(col_img, nbins=hist_bins)
			hog_feats = np.ravel(hog_feats)

			img_features = [];
			img_features.extend(bin_spatial_feats)
			img_features.extend(col_hist_feats)
			img_features.extend(hog_feats)
			
			#final_features = np.concatenate(img_features)

			# scaling
			final_features = self.scaler.transform(np.array(img_features).reshape(1, -1))

			# prediction
			pred = self.predict(final_features)
			
			if(pred==1):
				hot_windows.append(window)
		return hot_windows
"""					
	
