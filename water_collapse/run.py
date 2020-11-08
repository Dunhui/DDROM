import os
import sys
sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
import vtktools
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import keras.backend as K
import process_data as Pcd
import mkdirs
from keras.layers import Input, Dense, LSTM, Dropout
from keras import regularizers
from keras.models import Model, Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.callbacks import LearningRateScheduler

def variable_value():
	path =  "/home/ray/Downloads/fluidity-master/examples/water_collapse"
	vertify_rate = 0.4
	sequence_length = 50
	originalFile =  "/home/ray/Downloads/fluidity-master/examples/water_collapse_vertify" # original file
	destinationFile = "/home/ray/Downloads/fluidity-master/examples/water_collapse_predict" 

	return path, vertify_rate, sequence_length, originalFile, destinationFile

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

def predict_vol(dataset, predict_num, sequence_length):

	print(np.max(dataset),np.min(dataset), np.mean(dataset), np.median(dataset))

	encoder = load_model('vol_encoder.h5', compile = False) # encoder test data
	code = encoder.predict(dataset)
	print(np.max(code),np.min(code),np.mean(code), np.median(code), code.shape)

	LSTM = load_model('vol_lstm.h5', compile=False)
	outputs = predict_sequences_multiple(LSTM, code, sequence_length, predict_num)
	print(np.max(outputs),np.min(outputs),np.mean(outputs), np.median(outputs), outputs.shape)

	decoder = load_model('vol_decoder.h5', compile = False) # decoder predicted data
	predicted_vol = decoder.predict(outputs)
	print(np.max(predicted_vol),np.min(predicted_vol),np.mean(predicted_vol), np.median(predicted_vol))

	return predicted_vol

if __name__=="__main__":  

	path, vertify_rate, sequence_length, originalFile, destinationFile = variable_value()

	print("Data Preprocessing...")  
	vol = Pcd.get_data(path)
	dataset, vertify = Pcd.train_and_vertify(vol,vertify_rate) 
	scaler_data = MinMaxScaler()
	scaler_vol = scaler_data.fit_transform(dataset)

	print("Data Predicting...")  
	predicted_vol = predict_vol(scaler_vol,vertify.shape[0],sequence_length)
	scaler_outpus = scaler_data.inverse_transform(predicted_vol)

	value = mean_squared_error(vertify, scaler_outpus)
	print(value)

	mkdirs.mkdir(destinationFile)
	mkdirs.copyFiles(originalFile,destinationFile)
	vtu_num = Pcd.get_vtu_num(originalFile)
	mkdirs.transform(scaler_outpus, vtu_num, destinationFile)