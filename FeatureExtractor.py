import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from feature_utils import *
from sklearn.model_selection import train_test_split
import pickle

class FeatureExtractor():

	def __init__(self):
		self.non_vehicle_imgs = []
		self.vehicle_imgs = []
		self.non_vehicle_features = []
		self.vehicle_features = []
		self.training_features = []
		self.training_labels = []
		self.test_features = []
		self.test_labels = []
		self.bin_spatial_sh = None
		self.col_hist_sh = None
		self.hog_sh = None
	
	def load_data(self,nvdata,vdata):
		self.non_vehicle_imgs = nvdata
		self.vehicle_imgs = vdata

	def get_features(self):
		self.vehicle_features,self.bin_spatial_sh,self.col_hist_sh,self.hog_sh   = extract_features_YCrCb(self.vehicle_imgs, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=-1,
                        spatial_feat=True, hist_feat=True, hog_feat=True)

		self.non_vehicle_features,self.bin_spatial_sh,self.col_hist_sh,self.hog_sh   = extract_features_YCrCb(self.non_vehicle_imgs, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=-1,
                        spatial_feat=True, hist_feat=True, hog_feat=True)

	def get_training_testing_data(self):
		features = np.vstack((self.vehicle_features,self.non_vehicle_features)).astype(np.float64)

		labels = np.hstack((np.ones(len(self.vehicle_features)),np.zeros(len(self.non_vehicle_features))))

		rand_state = np.random.randint(0, 100)
		
		self.training_features, self.test_features, self.training_labels, self.test_labels = train_test_split(features, labels, test_size=0.2, random_state=rand_state)

	def pickle_data(self,filename):
		with open(filename,'wb') as f:
			pickle.dump(self.training_features,f)
			pickle.dump(self.training_labels,f)
			pickle.dump(self.test_features,f)
			pickle.dump(self.test_labels,f)

	def load_from_pickle(self,filename):
		with open(filename,'rb') as f:
			self.training_features = pickle.load(f)
			self.training_labels = pickle.load(f)
			self.test_features = pickle.load(f)
			self.test_labels = pickle.load(f)
			

		
		
	

		
