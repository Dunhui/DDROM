import os
import sys
sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import vtktools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, Flatten, Reshape
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler


		
class LoadmyData(object):
	"""docstring for LoadData"""
	def __init__(self):
		super(LoadmyData, self).__init__()
		
	def get_vtu_num(self,path):
	# count the number of vtu files
		f_list = os.listdir(path) 
		vtu_num = 0
		for i in f_list: 
			if os.path.splitext(i)[1] == '.vtu':
				vtu_num = vtu_num+1
		return vtu_num
              

	def get_velocity_data(self, path):
		
		if os.path.exists('/home/ray/Documents/github_code/circle/data/Velocity.npy'):
			velocity = np.load('/home/ray/Documents/github_code/circle/data/Velocity.npy')
		else:
			vtu_num = self.get_vtu_num(path)
			for n in range(vtu_num): 
				filename = path + "/circle-2d-drag_" + str(n)+ ".vtu" # set name of vtu files
				data = vtktools.vtu(filename)
				uvw = data.GetVectorField('Velocity')
				ui = np.hsplit(uvw,3)[0].T #velocity of x axis
				vi = np.hsplit(uvw,3)[1].T #velocity of y axis
				wi = np.hsplit(uvw,3)[2].T #velocity of z axis
				veli = np.hstack((ui,vi,wi)) #combine all into 1-d array
				vel = veli if n==0 else np.vstack((vel,veli))
			w = vel[:,int(vel.shape[1]/3)*2:]
			velocity = vel[:,:int(vel.shape[1]/3)*2] if np.all(w) == 0 else vel
			np.save('/home/ray/Documents/github_code/circle/data/Velocity.npy',velocity)
			
		print('The shape of \'Velocity\' is ',velocity.shape)
		
		return velocity

	def get_velocity_data_uvw(self, path, vtu_num):
	# get velocity data from vtu files 
		for n in range(vtu_num): 
			filename = path + "/circle-2d-drag_" + str(n)+ ".vtu" # set name of vtu files
			data = vtktools.vtu(filename)
			fieldList = data.GetFieldNames()
			Velocity_n = data.GetVectorField('Velocity') # shape = (points quantity,3)
			Velocity_n = np.reshape(Velocity_n,(1,Velocity_n.shape[0],Velocity_n.shape[1])) # reshape velocity data to (1,points quantity,3)
			velocity = Velocity_n if n==0 else np.vstack((velocity,Velocity_n))
		np.save('Velocity_uvw.npy',velocity)
		return velocity

	def get_cae_data(self, path):
	# call functions to load data
		if not os.path.exists(path):
			raise ValueError("The file {} does not exists".format(path)) # check the path of data
		vtu_num = self.get_vtu_num(path)

		Velocity = np.load('/home/ray/Documents/github_code/circle/data/Velocity_uvw.npy') \
		if os.path.exists('/home/ray/Documents/github_code/circle/data/Velocity_uvw.npy') \
		else self.get_velocity_data_uvw(path, vtu_num)


		# print('The shape of \'Velocity\' is ',Velocity.shape)
		scaler_u = MinMaxScaler()
		Velocity_scaler_u = scaler_u.fit_transform(Velocity[:,:,0])
		scaler_v = MinMaxScaler()
		Velocity_scaler_v = scaler_v.fit_transform(Velocity[:,:,1])
		scaler_w = MinMaxScaler()
		Velocity_scaler_w = scaler_w.fit_transform(Velocity[:,:,2])
		# print('\n scaler :',np.max(Velocity_scaler_u),np.min(Velocity_scaler_u), np.mean(Velocity_scaler_u), np.median(Velocity_scaler_u))
		Velocity_scaler_u = Velocity_scaler_u[:,:,np.newaxis]
		Velocity_scaler_v = Velocity_scaler_v[:,:,np.newaxis]
		Velocity_scaler_w = Velocity_scaler_w[:,:,np.newaxis]
		Velocity_scaler = np.dstack((Velocity_scaler_u, Velocity_scaler_v, Velocity_scaler_w))
		print('The shape of \'Velocity\' is ',Velocity_scaler.shape)
		return Velocity_scaler, scaler_u, scaler_v, scaler_w


	def train_and_test(self,dataset,test_rate):
	# divide dataset into train_dataset and test_dataset
	
		test_point = int(dataset.shape[0] * (1 - test_rate))
		train = dataset[:test_point,...]
		test = dataset[test_point:,...]

		return np.array(train), np.array(test)

	def create_dataset(self, dataset, look_back):

	    dataX, dataY = [], []
	    for i in range(len(dataset)-look_back-1):
	        a = dataset[i:(i+look_back),:]
	        dataX.append(a)
	        dataY.append(dataset[i + look_back,:])
	    Train_X = np.array(dataX)
	    Train_Y = np.array(dataY)

	    return Train_X, Train_Y
