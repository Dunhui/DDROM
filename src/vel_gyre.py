import os
import sys
sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
import vtktools
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model, Sequential, load_model

def path_value():
	path =  "/home/ray/Documents/data/gyre_2d"
	return path

def vel_variable_value():

	vertify_rate = 0.1
	my_epochs = 1000
	encoding_dim = 3
	# sequence_length = 24
	# LSTM_rate = 0.7
	# LSTM_epochs = 70 
	# predict_num = 960
	return vertify_rate, my_epochs, encoding_dim
	# return vertify_rate, my_epochs, encoding_dim, sequence_length, LSTM_rate, LSTM_epochs, predict_num

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
		filename = path + "/gyre_" + str(n)+ ".vtu"# name of vtu files
		data = vtktools.vtu(filename)
		uvw = data.GetVectorField('Velocity')
		ui = np.hsplit(uvw,3)[0].T #velocity of x axis
		vi = np.hsplit(uvw,3)[1].T #velocity of y axis
		wi = np.hsplit(uvw,3)[2].T #velocity of z axis
		veli = np.hstack((ui,vi,wi)) #combine all into 1-d array
		vel = veli if n==0 else np.vstack((vel,veli))
	w = vel[:,int(vel.shape[1]/3)*2:]
	outputs = vel[:,:int(vel.shape[1]/3)*2] if np.all(w) == 0 else vel
	np.save('gyre_Velocity.npy',outputs)
	print('The shape of \'Velocity\' is ',outputs.shape)

	return outputs
 
def get_data(path):

	vtu_num = get_vtu_num(path)
	velocity = np.load('gyre_Velocity.npy') if os.path.exists('gyre_Velocity.npy') else get_velocity_data(path, vtu_num)
	
	return velocity

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


def ae_velocity(vel, my_epochs, encoding_dim):# dim = 3

	input_img = Input(shape=(vel.shape[1], ))
	encoded = Dense(encoding_dim * 32, activation='relu')(input_img)
	encoded = Dense(encoding_dim * 16, activation='relu')(encoded)
	encoded = Dense(encoding_dim * 8, activation='relu')(encoded)
	encoded = Dense(encoding_dim * 4, activation='relu')(encoded)
	encoded = Dense(encoding_dim, activation='relu')(encoded)
	
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(encoding_dim * 4, activation='relu')(encoded)
	decoded = Dense(encoding_dim * 8, activation='relu')(decoded)
	decoded = Dense(encoding_dim * 16, activation='relu')(decoded)
	decoded = Dense(encoding_dim * 32, activation='relu')(decoded)
	decoded = Dense(vel.shape[1], activation='tanh')(decoded)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)

	# Separate Encoder model
	encoder = Model(input_img, encoded)
	encoded_input = Input(shape=(encoding_dim, ))

	# retrieve the layers of the autoencoder model
	# decoder_layer1 = autoencoder.layers[-3]
	# decoder_layer2 = autoencoder.layers[-2]
	# decoder_layer3 = autoencoder.layers[-1]

	# create the decoder model
	# decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

	# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
	autoencoder.compile(optimizer='Adagrad', loss = 'mean_absolute_percentage_error', metrics = ['accuracy'])

	# train the model
	x_train = vel
	# reduce_lr = LearningRateScheduler(scheduler)
	history = autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=1, shuffle=True, validation_split=0.2)
	draw_acc_loss(history)

	# save model
	# encoder.save('vel_encoder.h5') 
	# autoencoder.save('vel_ae.h5')
	# decoder.save('vel_decoder.h5')

	print("ae-model train succeed")  

def train_velocity_model(velocity):

	vertify_rate, my_epochs, encoding_dim = vel_variable_value()

	#vertify_rate, my_epochs, encoding_dim, sequence_length, LSTM_rate, LSTM_epochs, predict_num = vel_variable_value()

	dataset, vertify = train_and_vertify(velocity,vertify_rate) 
	print("training_dataset shape:",dataset.shape, "vertify_dataset shape:", vertify.shape, np.max(dataset), np.min(dataset))

	if not os.path.exists('gyre_vel_encoder.h5'):
		ae_velocity(dataset, my_epochs, encoding_dim)

	# if not os.path.exists('vel_lstm.h5'):
	# 	train, test = process_code_for_LSTM(dataset, 'vel_encoder.h5', sequence_length, LSTM_rate)
	# 	print(train.shape,test.shape)
	# 	LSTM_velocity(train, test, LSTM_epochs)

	# return dataset, vertify

if __name__=="__main__":  

	path = path_value()
	velocity = get_data(path)
	train_vel, vertify_vel = train_velocity_model(velocity)	

	