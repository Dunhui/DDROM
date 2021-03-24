import os
import vtktools
import joblib
import numpy as np
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
              
	def get_data(self, path, fileName, field_name, di, data_file_name, modelsFolder):
	# load data from　vtu files
		if not os.path.exists(path):
			raise ValueError("The file {} does not exists".format(path)) # check the path of data	
		
		data_npy_file = os.path.join(path, data_file_name) # set the name of npy data file

		if os.path.exists(data_npy_file):
			outputs = np.load(data_npy_file) # if data has already been loaded, load the npy file.
		else:
			vtu_num = self.get_vtu_num(path)# get the quantity of vtu files
			for i in range(vtu_num): 
				f_filename = path + fileName + str(i)+ ".vtu" # set name of vtu files
				data = vtktools.vtu(f_filename) # load data
				if di == 2:
					uvw = data.GetVectorField(field_name) # get Velocity data
					uvw = np.reshape(uvw[:,0:2],(1,uvw.shape[0],uvw.shape[1]-1)) # reshape velocity data to (1,points quantity,2)
					outputs = uvw if i==0 else np.vstack((outputs,uvw))# connect data 
				elif di == 1:
					output_i = data.GetScalarField(field_name) # get Velocity data
					outputs = output_i if i==0 else np.vstack((outputs,output_i))# connect data 
				else:
					print('the dimen of data has not been submitted')
			print('Data loaded from original vtu files.')
			np.save(data_npy_file,outputs) # save data
		
		print('Data loaded. The shape of ', field_name, ' is ',outputs.shape)

		if di == 2:
			scaler_u = MinMaxScaler() # data normalization
			Velocity_scaler_u = scaler_u.fit_transform(outputs[:,:,0])
			scaler_v = MinMaxScaler()
			Velocity_scaler_v = scaler_v.fit_transform(outputs[:,:,1])

			joblib.dump(scaler_u, modelsFolder + '/scaler_u.pkl')
			joblib.dump(scaler_v, modelsFolder + '/scaler_v.pkl')

			outputs_scaler = np.hstack((Velocity_scaler_u, Velocity_scaler_v))
			print('Data normalized, the shape of datset is :', outputs_scaler.shape)
		elif di == 1:
			scaler_1d = MinMaxScaler() # data normalization
			outputs_scaler = scaler_1d.fit_transform(outputs)	
			joblib.dump(scaler_1d, modelsFolder + '/scaler_1d.pkl')
		else:
			print('the dimen of data has not been submitted')	
		return outputs_scaler

	def data_shuffle(self,data):

		index = [i for i in range(len(data))]  
		np.random.shuffle(index) 
		data = data[index,:]

		return data

	def train_and_test(self,dataset,test_rate):
	# divide dataset into train_dataset and test_dataset
	
		test_point = int(dataset.shape[0] * (1 - test_rate))
		train = dataset[:test_point,...]
		test = dataset[test_point:,...]

		return np.array(train), np.array(test)

	def create_dataset(self, dataset, look_back):

	    data = []
	    for i in range(len(dataset)-look_back-1):
	        a = dataset[i:(i+look_back+1),...]
	        data.append(a)

	    data_group = np.array(data)
	    return data_group


	def divide_x_y(self, dataset):

		data_x = dataset[:,:-1,:]
		data_y = dataset[:,-1,:]
		return data_x, data_y

	
  #   def 2d_to_3d(self, 2d_data):
  #   	# add another zero dimensionality

		# x,y = 2d_data.shape[0], 2d_data.shape[1]
		# 3d_data = np.reshape(2d_data, (x, y, 1))
		# w_zero = np.zeros((x, y, 1))
		# 3d_data=np.concatenate((3d_data,w_zero), axis = 2)
		# print(3d_data.shape)

		# return 3d_data
