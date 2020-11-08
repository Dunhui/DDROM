import os
import sys
sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
import vtktools
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import keras.backend as K
import mkdirs
from keras.layers import Input, Dense, LSTM, Dropout
from keras import regularizers
from keras.models import Model, Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.callbacks import LearningRateScheduler


def variable_value():

	path =  "/home/ray/Documents/data/Rui_2002"
	vertify_rate = 0.4
	my_epochs = 300
	encoding_dim = 64
	originalFile = "/home/ray/Documents/data/Rui_2002_vertify"# original file
	destinationFile = "/home/ray/Documents/data/Rui_2002_test" # destination file

	return path, vertify_rate, my_epochs, encoding_dim, originalFile, destinationFile

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

	# divide dataset into train_dataset and vertify_dataset(70%,30%)
	vertify_point = int(dataset.shape[0] * (1 - vertify_rate))
	train = dataset[:vertify_point,:]
	vertify = dataset[vertify_point:,:]

	return np.array(train), np.array(vertify)

def draw_acc_loss(history):

	plt.figure(1)
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()


def ae_velocity(vel, my_epochs, encoding_dim):# dim = 3

	input_img = Input(shape=(vel.shape[1], ))
	encoded = Dense(encoding_dim * 32, activation='relu')(input_img)
	encoded = Dense(encoding_dim * 16, activation='relu')(encoded)
	encoded = Dense(encoding_dim * 8, activation='relu')(encoded)
	encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
	encoded = Dense(encoding_dim)(encoded)
	
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
	decoded = Dense(encoding_dim * 8, activation='relu')(decoded)
	decoded = Dense(encoding_dim * 16, activation='relu')(decoded)
	decoded = Dense(encoding_dim * 32, activation='relu')(decoded)
	decoded = Dense(vel.shape[1], activation='tanh')(decoded)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)
	encoder = Model(input_img, encoded)
	encoded_input = Input(shape=(encoding_dim, ))

	decoder_layer1 = autoencoder.layers[-1]
	decoder_layer2 = autoencoder.layers[-2]
	decoder_layer3 = autoencoder.layers[-3]
	decoder_layer4 = autoencoder.layers[-4]
	decoder_layer5 = autoencoder.layers[-5]

	# create the decoder model
	decoder = Model(encoded_input, decoder_layer1(decoder_layer2(decoder_layer3(decoder_layer4(decoder_layer5(encoded_input))))))

	# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
	autoencoder.compile(optimizer='adam', loss = 'mean_absolute_error', metrics = ['accuracy'])

	# train the model
	x_train = vel

	# def scheduler(epoch):
	# 	lr_epochs=10
	# 	lr = K.get_value(autoencoder.optimizer.lr)
	# 	K.set_value(autoencoder.optimizer.lr, lr * (0.1 ** (epoch // lr_epochs)))

	# 	return K.get_value(autoencoder.optimizer.lr)

	# reduce_lr = LearningRateScheduler(scheduler)

	# history = autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=128,  callbacks=[reduce_lr], validation_split=0.2)
	history = autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=64, validation_split=0.2)

	draw_acc_loss(history)

	encoder.save('vel_encoder.h5') 
	autoencoder.save('vel_ae.h5')
	decoder.save('vel_decoder.h5')

	print("ae-model train succeed")  

def train_velocity_model(velocity):
	my_epochs, encoding_dim = variable_value()[2],variable_value()[3]

	if not os.path.exists('vel_encoder.h5'):
		ae_velocity(velocity, my_epochs, encoding_dim)


if __name__=="__main__":  

	path, vertify_rate, my_epochs, encoding_dim, originalFile, destinationFile = variable_value()

	#load data
	print("Data loading...")
	velocity = get_data(path)
	dataset, vertify = train_and_vertify(velocity,vertify_rate) 
	print("training_dataset shape:",dataset.shape, "   vertify_dataset shape:", vertify.shape)

	#process data
	scaler = MinMaxScaler()
	scalered_velocity = scaler.fit_transform(dataset)

	#train model
	print("Model Building and training... ")
	# train_velocity_model(scalered_velocity)
	train_velocity_model(scalered_velocity)

	# test
	print("Data testing...")
	scaler_ver = MinMaxScaler()
	scaler_vertify = scaler_ver.fit_transform(vertify)

	# encoder = load_model('vel_encoder.h5', compile=False)
	# code = encoder.predict(scaler_vertify)
	
	# decoder = load_model('vel_decoder.h5', compile=False)
	# out_vertify = decoder.predict(code)
	# scaler_ver_output = scaler_ver.inverse_transform(out_vertify)	

	autoencoder = load_model('vel_ae.h5', compile=False)
	outpus  = autoencoder.predict(scaler_vertify)

	scaler_outpus = scaler_ver.inverse_transform(outpus)

	value = mean_squared_error(vertify, scaler_outpus)
	print(value)
	# mkdirs.transform(out_vertify, originalFile, destinationFile)








	