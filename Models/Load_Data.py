import os
import vtktools
import joblib
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

		
class Load_Data(object):
	"""docstring for LoadData"""
	def __init__(self):
		super(Load_Data, self).__init__()
		
	def get_vtu_num(self,path):
	# count the number of vtu files
		f_list = os.listdir(path) 
		vtu_num = 0
		for i in f_list: 
			if os.path.splitext(i)[1] == '.vtu':
				vtu_num = vtu_num+1
		return vtu_num
              
	def get_data(self, path, file_name, field_name, di, data_file_name, models_folder):
	# load data from　vtu files
		if not os.path.exists(path):
			raise ValueError("The file {} does not exists".format(path)) # check the path of data	

		if os.path.exists(data_file_name):
			outputs = np.load(data_file_name) # if data has already been loaded, load the npy file.
		else:
			vtu_num = self.get_vtu_num(path)# get the quantity of vtu files
			for i in range(vtu_num): 
				f_filename = path + file_name + str(i)+ ".vtu" # set name of vtu files
				data = vtktools.vtu(f_filename) # load data
				if di == 1:
					output_i = data.GetScalarField(field_name) # get Velocity data
					outputs = output_i if i==0 else np.vstack((outputs,output_i))# connect data 
				elif di == 2:
					uvw = data.GetVectorField(field_name) # get Velocity data
					uvw = np.reshape(uvw[:,0:2],(1,uvw.shape[0],uvw.shape[1]-1)) # reshape velocity data to (1,points quantity,2)
					outputs = uvw if i==0 else np.vstack((outputs,uvw))# connect data 
				elif di == 3:
					uvw = data.GetVectorField(field_name) # get data
					outputs = uvw if i==0 else np.vstack((outputs,uvw))# connect data 
				else:
					print('the dimen of data has not been submitted')
			# print('Data loaded from original vtu files.')
			np.save(data_file_name,outputs) # save data
		
		print('Data loaded. The shape of ', field_name, ' is ',outputs.shape)# , np.max(outputs), np.min(outputs))
		outputs = self.scaler_data(di, outputs, models_folder)
		print('Data normalized, the shape of dataset is :', outputs.shape)
		return outputs

	def scaler_data(self, di, outputs, models_folder, *ae_encoding_dim):

		if di == 1:
			scaler_1d = MinMaxScaler() # data normalization
			outputs = scaler_1d.fit_transform(outputs)	
			joblib.dump(scaler_1d, models_folder + '/scaler_1d.pkl')

		elif di == 2:
			scaler_u = MinMaxScaler() # data normalization
			Velocity_scaler_u = scaler_u.fit_transform(outputs[:,:,0])
			scaler_v = MinMaxScaler()
			Velocity_scaler_v = scaler_v.fit_transform(outputs[:,:,1])

			joblib.dump(scaler_u, models_folder + '/scaler_u.pkl')
			joblib.dump(scaler_v, models_folder + '/scaler_v.pkl')

			outputs = np.hstack((Velocity_scaler_u, Velocity_scaler_v))
		elif di == 3:
			scaler_u = MinMaxScaler() # data normalization
			Velocity_scaler_u = scaler_u.fit_transform(outputs[:,:,0])
			scaler_v = MinMaxScaler()
			Velocity_scaler_v = scaler_v.fit_transform(outputs[:,:,1])
			scaler_w = MinMaxScaler()
			Velocity_scaler_w = scaler_w.fit_transform(outputs[:,:,2])

			joblib.dump(scaler_u, models_folder + '/scaler_u.pkl')
			joblib.dump(scaler_v, models_folder + '/scaler_v.pkl')
			joblib.dump(scaler_w, models_folder + '/scaler_w.pkl')

			outputs = np.hstack((Velocity_scaler_u, Velocity_scaler_v, Velocity_scaler_w))
		elif di == 0:
			scaler_code = MinMaxScaler() # data normalization
			outputs = scaler_code.fit_transform(outputs)	
			ae_encoding_dim = ae_encoding_dim[0]
			joblib.dump(scaler_code, models_folder + '/scaler_code_' + str(ae_encoding_dim) + '.pkl')

		else:
			pass
		return outputs

	def scaler_inverse(self, di, outputs, models_folder, *ae_encoding_dim):
		# inverse_transform
		# di =1,2,3 is for AE model, di=0 is for attention codes or POD codes
		if di == 1:
			scaler = joblib.load(models_folder + '/scaler_1d.pkl')
			outputs = scaler.inverse_transform(outputs) 
		elif di == 2:
			u ,v = np.hsplit(outputs, 2)
			# u = outputs[:,:,0]
			# v = outputs[:,:,1]
			scaler_u = joblib.load(models_folder + '/scaler_u.pkl')
			scaler_v = joblib.load(models_folder + '/scaler_v.pkl')
			outputs_u = scaler_u.inverse_transform(u)
			outputs_v = scaler_v.inverse_transform(v)
			outputs = np.dstack((outputs_u, outputs_v))
		elif di == 3:
			u ,v, w = np.hsplit(outputs, 3)
			scaler_u = joblib.load(models_folder + '/scaler_u.pkl')
			scaler_v = joblib.load(models_folder + '/scaler_v.pkl')
			scaler_w = joblib.load(models_folder + '/scaler_w.pkl')
			outputs_u = scaler_u.inverse_transform(u)
			outputs_v = scaler_v.inverse_transform(v)
			outputs_w = scaler_w.inverse_transform(w)
			outputs = np.dstack((outputs_u, outputs_v, outputs_w))
		elif di == 0:
			ae_encoding_dim = ae_encoding_dim[0]
			scaler = joblib.load(models_folder + '/scaler_code_' + str(ae_encoding_dim) + '.pkl')
			outputs = scaler.inverse_transform(outputs) 
		else:
			pass
			
		return outputs

	def create_dataset(self, dataset, look_back):

	    data = []
	    for i in range(len(dataset)-look_back-1):
	        data.append(dataset[i:(i+look_back+1),...])# data start from [0], [128]
	    # data_group = np.array(data)
	    return np.array(data)

def magnitude(data):
# return value of (x^2+y^2+z^0)^0.5
	output = []
	for i in range(data.shape[0]):
		output.append(np.sqrt(np.sum(np.square(data[i]), axis = 1)))
	return np.array(output)
	
