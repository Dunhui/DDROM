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
from keras.callbacks import ReduceLROnPlateau


def variable_value():

	path =  "/home/ray/Documents/data/Rui_2002"
	vertify_rate = 0.4
	sequence_length = 20
	LSTM_rate = 0.8
	LSTM_epochs = 50 

	return path, vertify_rate, sequence_length, LSTM_rate, LSTM_epochs

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

def get_data(path):

	vtu_num = get_vtu_num(path)
	velocity = np.load('Velocity.npy') if os.path.exists('Velocity.npy') else get_velocity_data(path, vtu_num)

	return velocity


def train_and_vertify(dataset,vertify_rate):

	# divide dataset into train_dataset and vertify_dataset(60%,40%)
	vertify_point = int(dataset.shape[0] * (1 - vertify_rate))
	train = dataset[:vertify_point,:]
	vertify = dataset[vertify_point:,:]

	return np.array(train), np.array(vertify)
 
def draw_acc_loss(history):

	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

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


def LSTM_velocity(train, test, LSTM_epochs):
	
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
	draw_acc_loss(history)
	# save model
	model.save('vel_lstm.h5')
    
	testYPredict = model.predict(testX)
	trainScore = math.sqrt(mean_absolute_error(testY, testYPredict))
	print("LSTM-model train succeed  ", 'Train Score: %.6f MSE' % (trainScore)) 


def train_velocity_model(velocity):

	path, vertify_rate, sequence_length, LSTM_rate, LSTM_epochs = variable_value()

	if not os.path.exists('vel_lstm.h5'):
		encoder_model = '/home/ray/Documents/ae-for-cfd-master/src/Autoencoder/vel_encoder.h5'
		train, test = process_code_for_LSTM(velocity, encoder_model, sequence_length, LSTM_rate)
		LSTM_velocity(train, test, LSTM_epochs)

if __name__=="__main__":  

	path, vertify_rate, sequence_length, LSTM_rate, LSTM_epochs = variable_value()

	print("Data loading...")
	velocity = get_data(path)
	dataset, vertify = train_and_vertify(velocity,vertify_rate) 
	print("training_dataset shape:",dataset.shape, "   vertify_dataset shape:", vertify.shape)
	scaler = MinMaxScaler()
	scalered_velocity = scaler.fit_transform(dataset)

	print("Model Building and training... ")
	train_velocity_model(scalered_velocity)