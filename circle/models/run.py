import os
import sys
sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
import numpy as np
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import glorot_uniform
from sklearn.preprocessing import MinMaxScaler

import mkdirs_trans
import FCT_model

import AE_model
from LoadVolData import *



def predict_velocity(dataset, predict_num):
	print(np.max(dataset),np.min(dataset))
	sequence_length = 20

	encoder = load_model('/home/ray/Documents/github_code/circle/code/saved_models/DeepAE_vel_encoder.h5', compile = False) # encoder test data
	code = encoder.predict(dataset)
	print(np.max(code),np.min(code),code.shape)

	middle_LSTM = load_model('/home/ray/Documents/github_code/circle/code/saved_models/vel_multiAtten.h5', compile=False)
	outputs = predict_sequences_multiple(middle_LSTM, code, sequence_length, predict_num)
	# 原压缩后code没有负值，但预测后出现负值
	print(np.max(outputs),np.min(outputs), outputs.shape)


	decoder = load_model('/home/ray/Documents/github_code/circle/code/saved_models/DeepAE_vel_decoder.h5', compile = False) # decoder predicted data
	predicted_vel = decoder.predict(outputs)

	print(np.max(predicted_vel),np.min(predicted_vel))

	return predicted_vel

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


if __name__=="__main__":  

	# ROM train
	path =  "/home/ray/Documents/github_code/circle/data/Rui_2002"
	originalFile = '/home/ray/Documents/github_code/circle/data/Rui_2002'
	ROM(path, originalFile)

	# FCT train
    data_ae = np.load("/home/ray/Documents/github_code/circle/data/AE_Code_for_Predict")
    data_deepae = np.load("/home/ray/Documents/github_code/circle/data/Deep_Code_for_Predict")
    data_cae = np.load("/home/ray/Documents/github_code/circle/data/CAE_Code_for_Predict")
    print(data_ae.shape, data_deepae.shape, data_cae.shape)
    FCT(data_deepae)

    # whole process
	print("Data loading...")
	data = LoadmyData()
	vol= data.get_velocity_data(path)
	scaler = MinMaxScaler(feature_range=(0, 1))
	Velocity_scalered = scaler.fit_transform(vol)
	train, test = data.train_and_test(vol, test_rate = 0.5)
	print(test.shape)
	predict_velocity(train, test.shape[0])




