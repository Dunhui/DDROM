import os
import sys
sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
import vtktools
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import keras.backend as K
from keras.layers import Input, Dense, LSTM, Dropout
from keras import regularizers
from keras.models import Model, Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.callbacks import LearningRateScheduler


def variable_value():

	path =  "/home/ray/Documents/data/Rui_2002"
	vertify_rate = 0.4
	my_epochs = 50
	encoding_dim = 3

	return path, vertify_rate, my_epochs, encoding_dim

def get_vtu_num(path):
# count the number of vtu files
	f_list = os.listdir(path) 
	vtu_num = 0
	for i in f_list:
		if os.path.splitext(i)[1] == '.vtu':
			vtu_num = vtu_num+1

	return vtu_num

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
	pressure = np.load('Pressure.npy') if os.path.exists('Pressure.npy') else get_pressure_data(path, vtu_num)

	return pressure


def train_and_vertify(dataset,vertify_rate):

	# divide dataset into train_dataset and vertify_dataset(70%,30%)
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

def ae_pressure(dataset,my_epochs, encoding_dim):

	input_img = Input(shape=(dataset.shape[1], ))

	encoded = Dense(encoding_dim * 32, activation='relu')(input_img)
	encoded = Dense(encoding_dim * 16, activation='relu')(encoded)
	encoded = Dense(encoding_dim * 8, activation='relu')(encoded)
	encoded = Dense(encoding_dim)(encoded)
	
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(encoding_dim * 8, activation='relu')(encoded)
	decoded = Dense(encoding_dim * 16, activation='relu')(decoded)
	decoded = Dense(encoding_dim * 32, activation='relu')(decoded)
	decoded = Dense(dataset.shape[1], activation='tanh')(decoded)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)

	# Separate Encoder model
	encoder = Model(input_img, encoded)
	encoded_input = Input(shape=(encoding_dim, ))

	# retrieve the layers of the autoencoder model
	# decoder_layer0 = autoencoder.layers[-4]
	decoder_layer1 = autoencoder.layers[-1]
	decoder_layer2 = autoencoder.layers[-2]
	decoder_layer3 = autoencoder.layers[-3]
	decoder_layer4 = autoencoder.layers[-4]

	# create the decoder model
	decoder = Model(encoded_input, decoder_layer1(decoder_layer2(decoder_layer3(decoder_layer4(encoded_input)))))

	# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
	autoencoder.compile(optimizer='adam', loss = 'mean_absolute_error', metrics = ['accuracy'])

	# train the model
	x_train = dataset

	def scheduler(epoch):
		lr_epochs=20
		lr = K.get_value(autoencoder.optimizer.lr)
		K.set_value(autoencoder.optimizer.lr, lr * (0.1 ** (epoch // lr_epochs)))

		return K.get_value(autoencoder.optimizer.lr)

	reduce_lr = LearningRateScheduler(scheduler)

	history = autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=8, shuffle=True, validation_split=0.2, callbacks=[reduce_lr])
	
	draw_acc_loss(history)

	# save model
	encoder.save('pre_encoder.h5') 
	autoencoder.save('pre_ae.h5')
	decoder.save('pre_decoder.h5')

	print("ae-model train succeed")  


def train_pressure_model(velocity):
	my_epochs, encoding_dim = variable_value()[2],variable_value()[3]

	if not os.path.exists('pre_ae.h5'):
		ae_pressure(velocity, my_epochs, encoding_dim)


if __name__=="__main__":  

	path, vertify_rate, my_epochs, encoding_dim = variable_value()

	print("Data loading...")
	pressure = get_data(path)
	dataset, vertify = train_and_vertify(pressure,vertify_rate) 
	print("training_dataset shape:",dataset.shape, "   vertify_dataset shape:", vertify.shape)
	scaler = MinMaxScaler()
	scalered_pressure = scaler.fit_transform(dataset)

	print("Model Building and training... ")
	train_pressure_model(scalered_pressure)
