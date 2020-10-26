import os
import sys
sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
import vtktools
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import keras.backend as K
import shutil
from keras.layers import Input, Dense, LSTM, Dropout
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error

def path_value():
	path =  "/home/ray/Documents/data/Rui_2002"
	originalFile = "/home/ray/Documents/data/Rui_2002_vertify"# original file
	destinationFile = "/home/ray/Documents/data/predicted" # destination file

	return path, originalFile, destinationFile

def variable_value():

	vertify_rate = 0.4
	sequence_length = 20
	predict_num = 960

	return vertify_rate, sequence_length, predict_num


def get_vtu_num(path):
# count the number of vtu files
	f_list = os.listdir(path) 
	vtu_num = 0
	for i in f_list:
		if os.path.splitext(i)[1] == '.vtu':
			vtu_num = vtu_num+1
	return vtu_num

def get_velocity_data(path, vtu_num):

	for n in range(vtu_num): 
		filename = path + "/circle-2d-drag_" + str(n)+ ".vtu"# name of vtu files
		data = vtktools.vtu(filename)
		uvw = data.GetVectorField('Velocity')
		ui = np.hsplit(uvw,3)[0].T #velocity of x axis
		vi = np.hsplit(uvw,3)[1].T #velocity of y axis
		wi = np.hsplit(uvw,3)[2].T #velocity of z axis
		veli = np.hstack((ui,vi,wi)) #combine all into 1-d array
		vel = veli if n==0 else np.vstack((vel,veli))
	w = vel[:,int(vel.shape[1]/3)*2:]
	outputs = vel[:,:int(vel.shape[1]/3)*2] if np.all(w) == 0 else vel
	np.save('Velocity.npy',outputs)
	print('The shape of \'Velocity\' is ',outputs.shape)

	return outputs

def get_pressure_data(path, vtu_num):
	
	for n in range(vtu_num): 
		filename = path + "/circle-2d-drag_" + str(n)+ ".vtu"# name of vtu files
		data = vtktools.vtu(filename)
		pressure_i = data.GetScalarField('Pressure')
		pressure = pressure_i if n==0 else np.vstack((pressure,pressure_i))
	np.save('Pressure.npy',pressure)
	print('The shape of \'Pressure\' is ',pressure.shape)

	return pressure

def get_data(path):

	vtu_num = get_vtu_num(path)
	velocity = np.load('Velocity.npy') if os.path.exists('Velocity.npy') else get_velocity_data(path, vtu_num)
	pressure = np.load('Pressure.npy') if os.path.exists('Pressure.npy') else get_pressure_data(path, vtu_num)

	return velocity, pressure


def train_and_vertify(dataset,vertify_rate):

	# divide dataset into train_dataset and vertify_dataset(70%,30%)
	vertify_point = int(dataset.shape[0] * (1 - vertify_rate))
	train = dataset[:vertify_point,:]
	vertify = dataset[vertify_point:,:]

	return np.array(train), np.array(vertify)
 

def predict_sequences_multiple(model, data, sequence_length, predict_num):
	
	data_origin = data[data.shape[0] - sequence_length + 1:,:]
	data = data_origin.reshape(1,data_origin.shape[0],data_origin.shape[1]) 

	print('[Model] Predicting Sequences Multiple...')
	for i in range(predict_num):
		list_p = data[:,:,:] if i == 0 else data[:,i:,:]
		code = model.predict(list_p)
		code = code.reshape(1,1,code.shape[1])
		data = np.concatenate((data,code), axis = 1) 
	data = data.reshape(data.shape[1],data.shape[2])
	return data[sequence_length-1:,:]


# copy original data
def copyFiles(sourceDir,targetDir):
    if sourceDir.find("exceptionfolder")>0:
        return

    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir,file)
        targetFile = os.path.join(targetDir,file)

        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or (os.path.exists(targetFile) and (os.path.getsize(targetFile) !=os.path.getsize(sourceFile))):
                open(targetFile, "wb").write(open(sourceFile, "rb").read())
                # print(targetFile+" copy succeeded")

        if os.path.isdir(sourceFile):
            copyFiles(sourceFile, targetFile)

# # create new folder
def mkdir(path):
	folder = os.path.exists(path)

	if folder:                   
		print ("---  We already have this folder name  ---")
		shutil.rmtree(path, ignore_errors=True)
		print ("---  We already delete this folder  ---")

	print ("---  create new folder...---")
	os.makedirs(path)
	("---  OK  ---")

def transform(predicted_vel, originalFile, destinationFile):

	mkdir(destinationFile)     
	copyFiles(originalFile,destinationFile)

# 	# replace velocity with new output data 
	for i in range(predicted_vel.shape[0]):
		f_filename=destinationFile + "/circle-2d-drag_" + str(i+1201)+ ".vtu"
		f_file = vtktools.vtu(f_filename) 
		velocity_uv = predicted_vel[i].reshape((2,int(predicted_vel.shape[1]/2)))
		w = np.zeros(velocity_uv.shape[1]).reshape((1,velocity_uv.shape[1]))
		velocity_uvw = np.vstack((velocity_uv,w)).T
		f_file.AddVectorField("Velocity_dim", velocity_uvw)
		# f_file.AddScalarField("Pressure_predict", predicted_pre[i])
		f_file.Write(f_filename)
	
	print('transform succeed')	

def predict_velocity(dataset, predict_num):
	
	sequence_length = vel_variable_value()[1]

	encoder = load_model('/home/ray/Documents/ae-for-cfd-master/src/Autoencoder/vel_encoder.h5', compile = False) # encoder test data
	code = encoder.predict(dataset)

	LSTM = load_model('/home/ray/Documents/ae-for-cfd-master/src/LSTM/vel_lstm.h5', compile=False)
	outputs = predict_sequences_multiple(LSTM, code, sequence_length, predict_num)

	decoder = load_model('/home/ray/Documents/ae-for-cfd-master/src/Autoencoder/vel_decoder.h5', compile = False) # decoder predicted data
	predicted_vel = decoder.predict(outputs)

	return predicted_vel

def predict_pressure(dataset, predict_num):

	sequence_length = variable_value()[1]

	encoder = load_model('/home/ray/Documents/ae-for-cfd-master/src/Autoencoder/pre_encoder.h5', compile = False) # encoder test data
	code = encoder.predict(dataset)

	LSTM = load_model('/home/ray/Documents/ae-for-cfd-master/src/LSTM/pre_lstm.h5', compile=False)
	outputs = predict_sequences_multiple(LSTM, code, sequence_length, predict_num)

	decoder = load_model('/home/ray/Documents/ae-for-cfd-master/src/Autoencoder/pre_decoder.h5', compile = False) # decoder predicted data
	predicted_pre = decoder.predict(outputs)

	return predicted_pre

if __name__=="__main__":  

	path, originalFile, destinationFile = path_value()
	vertify_rate, sequence_length, predict_num = variable_value()

	print("Data loading...")
	velocity, pressure = get_data(path)
	vel_dataset, vel_vertify = train_and_vertify(velocity,vertify_rate) 
	pre_dataset, pre_vertify = train_and_vertify(pressure,vertify_rate) 
	print("velocity training_dataset shape:",vel_dataset.shape, "   vertify_dataset shape:", vel_vertify.shape)
	print("pressure training_dataset shape:",pre_dataset.shape, "   vertify_dataset shape:", pre_vertify.shape)
	scaler_1 = MinMaxScaler()
	scalered_velocity = scaler_1.fit_transform(vel_dataset)
	scaler_2 = MinMaxScaler()
	scalered_pressure = scaler_2.fit_transform(pre_dataset)

	print("Model predicting...")

	predicted_vel = predict_velocity(scalered_velocity, vel_vertify.shape[0])
	predicted_vel = scaler_1.inverse_transform(predicted_vel) 
	print(predicted_vel.shape)

	predicted_pre = predict_pressure(scalered_pressure, pre_vertify.shape[0])
	predicted_pre = scaler_2.inverse_transform(predicted_pre) 
	print(predicted_pre.shape)

	transform(predicted_vel, predicted_pre, originalFile, destinationFile)
