import os
import sys
sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
import vtktools
import math
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import process_data as Pcd
from keras.layers import Input, Dense, LSTM, Dropout
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.callbacks import ReduceLROnPlateau


def variable_value():

	path =  "/home/ray/Downloads/fluidity-master/examples/water_collapse"
	vertify_rate = 0.4
	sequence_length = 50
	LSTM_rate = 0.8
	LSTM_epochs = 50 

	return path, vertify_rate, sequence_length, LSTM_rate, LSTM_epochs


def process_code_for_LSTM(dataset, model_name, sequence_length, LSTM_rate):

	encoder = load_model(model_name, compile=False)
	code = encoder.predict(dataset)
	np.save('Code_for_lstm.npy',code)
	print('The shape of \'Code for Lstm\' is ',code.shape)

	data = []
	for i in range(len(code) - sequence_length + 1):
		data.append(code[i: i + sequence_length,:])
		reshaped_data = np.array(data).astype('float64')

	# divide dataset into train data and test data(80%,20%)
	train_size = int(reshaped_data.shape[0] * LSTM_rate) 
	test_size = reshaped_data.shape[0] - train_size
	train, test = reshaped_data[0:train_size,:,:], reshaped_data[train_size:len(code),:,:]
	return train, test


# divide dataset into data and label
def divide_data_X_Y(reshaped_data): 
	dataX, dataY = [], []
	dataX = reshaped_data[:,:-1,:]
	dataY = reshaped_data[:,-1,:]
	return np.array(dataX), np.array(dataY)


def LSTM_pressure(train, test, LSTM_epochs):
	
	#divide dataset into data and label
	trainX, trainY = divide_data_X_Y(train)          
	testX, testY = divide_data_X_Y(test)

	# model structure
	model = Sequential()
	model.add(LSTM(50, return_sequences=True))
	model.add(LSTM(100, return_sequences=False))
	model.add(Dense(trainX.shape[2]))

	# compile model
	model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
	
	# train model
	history = model.fit(trainX, trainY, epochs=LSTM_epochs, batch_size=64, validation_data = (testX,testY),verbose = 2)
	Pcd.draw_acc_loss(history)
	# save model
	model.save('vol_lstm.h5')
    
	testYPredict = model.predict(testX)
	trainScore = math.sqrt(mean_absolute_error(testY, testYPredict))
	print("LSTM-model train succeed  ", 'Train Score: %.6f MSE' % (trainScore)) 


def train_lstm_model(scalered_vol):

	path, vertify_rate, sequence_length, LSTM_rate, LSTM_epochs = variable_value()

	if not os.path.exists('vol_lstm.h5'):
		encoder_model = '/home/ray/Documents/github_rep/water_collapse/vol_encoder.h5'
		train, test = process_code_for_LSTM(scalered_vol, encoder_model, sequence_length, LSTM_rate)
		LSTM_pressure(train, test, LSTM_epochs)

if __name__=="__main__":  

	path, vertify_rate, sequence_length, LSTM_rate, LSTM_epochs = variable_value()

	print("Data loading...")
	vol = Pcd.get_data(path)
	dataset, vertify = Pcd.train_and_vertify(vol,vertify_rate) 
	print("training_dataset shape:",dataset.shape, "   vertify_dataset shape:", vertify.shape)
	scaler = MinMaxScaler()
	scalered_vol = scaler.fit_transform(dataset)

	print("Model Building and training... ")
	train_lstm_model(scalered_vol)