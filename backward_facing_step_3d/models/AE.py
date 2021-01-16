import os
import sys
sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, Flatten, Reshape
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from modelProcessing import save_model, draw_Acc_Loss
from mkdirs import transform_vector
from LoadData import *

class AE(object):
	"""Autoencoder Model"""
	def __init__(self):
		super(AE, self).__init__()
	
	def build_AE_model(self, encoding_dim):

		input_img = Input(shape=(191388, ))
		encoded = Dense(encoding_dim , activation='relu')(input_img)
		decoded = Dense(191388, activation='tanh')(encoded)

		self.autoencoder = Model(input_img, decoded)
		self.encoder = Model(input_img, encoded)
		encoded_input = Input(shape=(encoding_dim,))
		decoder_layer = self.autoencoder.layers[-1]
		self.decoder = Model(encoded_input, decoder_layer(encoded_input))

		self.autoencoder.compile(optimizer='adam', loss = 'mse', metrics = ['accuracy'])
		self.autoencoder.summary()

	def train_AE_model(self, x_train, x_test, epochs, batch_size):

		save_dir = os.path.join(os.getcwd(), 'saved_models')
		filepath="AEweights.h5"	# OR filepath="model_{epoch:02d}-{loss:.2f}.h5"
		checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='loss', verbose=1, save_best_only=True, mode='min')

		self.history_record = self.autoencoder.fit(x_train, x_train, epochs = epochs, batch_size = batch_size, validation_split=0.2, callbacks = [checkpoint])
		
		draw_Acc_Loss(self.history_record)		
		save_model(self.encoder, 'AE_vel_encoder.h5', save_dir)
		save_model(self.decoder, 'AE_vel_decoder.h5', save_dir)
		save_model(self.autoencoder, 'AE_vel.h5', save_dir)

		print(" AE model for Velocity trained successfully")  

		scores = self.autoencoder.evaluate(x_test, x_test, verbose=1)
		print('Test loss:', scores[0], '\nTest accuracy:', scores[1])
	

class DeepAE(object):
	"""DeepAutoencoder Model"""
	def __init__(self):
		super(deepAE, self).__init__()

	def build_DeepAE_model(self, encoding_dim):

		input_img = Input(shape=(95694*3, ))
		encoded = Dense(encoding_dim , activation='relu', name='encoded')(input_img)
		encoded = Dense(encoding_dim * 16, activation='relu')(encoded)
		encoded = Dense(encoding_dim * 8, activation='relu')(encoded)
		encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
		encoded = Dense(encoding_dim)(encoded)

		# "decoded" is the lossy reconstruction of the input
		decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
		decoded = Dense(encoding_dim * 8, activation='relu')(decoded)
		decoded = Dense(encoding_dim * 16, activation='relu')(decoded)
		decoded = Dense(vel.shape[1], activation='tanh')(decoded)

		# this model maps an input to its reconstruction
		self.autoencoder = Model(input_img, decoded)
		self.encoder = Model(input_img, encoded)
		encoded_input = Input(shape=(encoding_dim, ))

		decoder_layer1 = autoencoder.layers[-1]
		decoder_layer2 = autoencoder.layers[-2]
		decoder_layer3 = autoencoder.layers[-3]
		decoder_layer4 = autoencoder.layers[-4]
		decoder_layer5 = autoencoder.layers[-5]

		# create the decoder model
		self.decoder = Model(encoded_input, decoder_layer1(decoder_layer2(decoder_layer3(decoder_layer4(decoder_layer5(encoded_input))))))

		# def contractive_loss(y_pred, y_true):
		# 	lam = 1e-4
		# 	mse = K.mean(K.square(y_true - y_pred), axis=1)

		# 	W = K.variable(value=autoencoder.get_layer('encoded').get_weights()[0])  # N x N_hidden
		# 	W = K.transpose(W)  # N_hidden x N
		# 	h = autoencoder.get_layer('encoded').output
		# 	dh = h * (1 - h)  # N_batch x N_hidden

		# 	# N_batch x N_hidden * N_hidden x 1 = N_batch x 1
		# 	contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

		# 	return mse + contractive

		# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
		self.autoencoder.compile(optimizer='adam', loss = 'mse', metrics = ['accuracy'])
		# self.autoencoder.compile(optimizer='adam', loss = contractive_loss, metrics = ['accuracy'])
		self.autoencoder.summary()

		
	def train_DeepAE_model(self, x_train, x_test, epochs, batch_size):

		save_dir = os.path.join(os.getcwd(), 'saved_models')
		filepath="DeepAEweights.h5"	# OR filepath="model_{epoch:02d}-{loss:.2f}.h5"
		checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='loss', verbose=1, save_best_only=True, mode='min')

		self.history_record = self.autoencoder.fit(x_train, x_train, epochs = epochs, batch_size = batch_size, validation_split=0.2, callbacks = [checkpoint])
		
		draw_Acc_Loss(self.history_record)		
		save_model(self.encoder, 'DeepAE_vel_encoder.h5', save_dir)
		save_model(self.decoder, 'DeepAE_vel_decoder.h5', save_dir)
		save_model(self.autoencoder, 'DeepAE_vel.h5', save_dir)

		print(" DeepAE model trained successfully")  

		scores = self.autoencoder.evaluate(x_test, x_test, verbose=1)
		print('Test loss:', scores[0], '\nTest accuracy:', scores[1])
	

						
class CAE(object):
	# the Convolutional Autoencoder Model
	def __init__(self):
		super(CAE, self).__init__()
		
	def process_data_for_CAE(self, path):
		#  Load data
		print("Data loading for CAE...")
		data = LoadData()
		scaler_vol, scaler_u, scaler_v, scaler_w  = data.get_cae_data(path) #Velocity_scaler, scaler_u, scaler_v, scaler_w

		# split out the training set and test set.
		train, test = data.train_and_test(scaler_vol, test_rate=0.2) # (1136, 95694, 3)&(487, 95694, 3)
		train = train.reshape(train.shape[0], train.shape[1], train.shape[2], 1)
		test = test.reshape(test.shape[0], test.shape[1], test.shape[2], 1)
		print("train_dataset shape:",train.shape, "   test_dataset shape:", test.shape) # (1136, 95694, 3, 1)&(487, 95694, 3, 1)

		return scaler_vol, train, test, scaler_u, scaler_v, scaler_w

	def build_CAE_model(self, encoding_dim):
	# model layers

		# input placeholder
		CHANNEL_1 = 64
		CHANNEL_2 = 32
		CHANNEL_3 = 16
		CHANNEL_OUTPUT=1
		input_image = Input(shape=(95694, 3, 1))

		# encoding layer
		x = Conv2D(CHANNEL_1, (3, 3), activation='relu',padding="same")(input_image)
		x = MaxPool2D((6, 1), padding='same')(x)
		x = Conv2D(CHANNEL_2, (3, 3), activation='relu', padding='same')(x)
		x = MaxPool2D((41, 1), padding='same')(x)
		x = Conv2D(CHANNEL_3, (3, 3), activation='relu', padding='same')(x)
		encode_output = MaxPool2D((389, 1), padding='same')(x)

		# build surprised learning model
		encode_output_flatten = Flatten()(encode_output)
		core = Dense(encoding_dim, activation='softmax')(encode_output_flatten)
		SL_output = Dense(48, activation='relu')(core)
		encode_output = Reshape((1, 3, CHANNEL_3))(SL_output)

		# decoding layer
		x = Conv2D(CHANNEL_1, (3, 3), activation='relu',padding='same')(encode_output)
		x = UpSampling2D((389, 1))(x)
		x = Conv2D(CHANNEL_2, (3, 3), activation='relu',padding='same')(x)
		x = UpSampling2D((41, 1))(x)
		x = Conv2D(CHANNEL_1, (3, 3), activation='relu', padding='same')(x)
		x = UpSampling2D((6, 1))(x)
		decode_output = Conv2D(CHANNEL_OUTPUT, (3, 3),activation='relu', padding='same')(x)

		# build autoencoder, encoder
		autoencoder = Model(inputs=input_image, outputs=decode_output)
		encoder = Model(inputs=input_image, outputs=core)

		# build decoder
		encoded_input = Input(shape=(encoding_dim,))
		decoder_layer1 = autoencoder.layers[-1]
		decoder_layer2 = autoencoder.layers[-2]
		decoder_layer3 = autoencoder.layers[-3]
		decoder_layer4 = autoencoder.layers[-4]
		decoder_layer5 = autoencoder.layers[-5]
		decoder_layer6 = autoencoder.layers[-6]
		decoder_layer7 = autoencoder.layers[-7]
		decoder_layer8 = autoencoder.layers[-8]
		decoder_layer9 = autoencoder.layers[-9]

		# create the decoder model
		# decoder = Model(encoded_input, decoder_layer1(decoder_layer2(decoder_layer3(decoder_layer4(decoder_layer5(decoder_layer6(decoder_layer7(encoded_input))))))))
		decoder = Model(encoded_input, decoder_layer1(decoder_layer2(decoder_layer3(decoder_layer4(decoder_layer5(decoder_layer6(decoder_layer7(decoder_layer8(decoder_layer9(encoded_input))))))))))
		
		# compile autoencoder
		autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
		
		autoencoder.summary()

		self.autoencoder = autoencoder
		self.encoder = encoder
		self.decoder = decoder	

	def train_CAE_model(self, x, x_test, epochs, batch_size):
	# set checkpoint, train model, save model and weight 

		save_dir = os.path.join(os.getcwd(), 'saved_models')
		filepath="CAEweights.h5"	# OR filepath="model_{epoch:02d}-{loss:.2f}.h5"
		checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='loss', verbose=1, save_best_only=True, mode='min')

		self.history_record = self.autoencoder.fit(x, x, epochs = epochs, batch_size = batch_size, validation_split=0.2, callbacks = [checkpoint])
		
		draw_Acc_Loss(self.history_record)		
		save_model(self.encoder, 'CAE_vel_encoder.h5', save_dir)
		save_model(self.decoder, 'CAE_vel_decoder.h5', save_dir)
		save_model(self.autoencoder, 'CAE_vel.h5', save_dir)

		print(" CAE model trained successfully")  

		scores = self.autoencoder.evaluate(x_test, x_test, verbose=1)
		print('Test loss:', scores[0], '\nTest accuracy:', scores[1])
	

if __name__=="__main__":  

	# set variables  	 
	path =  "/home/ray/Downloads/software/fluidity-master/examples/backward_facing_step_3d"
	originalFile = '/home/ray/Downloads/software/fluidity-master/examples/backward_facing_step_3d'
	
	#  Load data for AE & DeepAE
	print("Data loading...")
	data = LoadData()
	vol= data.get_velocity_data(path)
	train, test = data.train_and_test(vol, test_rate = 0.2)

	# build & train model
	# AE
	ae = AE()
	ae.build_AE_model(encoding_dim = 9)
	ae.train_AE_model(train, test, epochs = 200, batch_size = 24)
	# test
	ae_model = '/home/ray/Documents/github_code/backward_facing_step_3d/models/saved_models/AE_vel.h5'
	ae = load_model(ae_model, compile=False)
	ae_outputs = ae.predict(vol)
	print('The shape of \'AE outputs\' is ',ae_outputs.shape, '\nStart to update data in vtu files...')
	destinationFile = '/home/ray/Documents/github_code/backward_facing_step_3d/data/AE_outputs'
	transform_vector(ae_outputs, ae_outputs.shape[0], originalFile, destinationFile)

	# DeepAE
	deepAE = DeepAE()
	deepAE.build_DeepAE_model(encoding_dim = 9)
	deepAE.train_DeepAE_model(train, test, epochs = 15, batch_size = 12)
	# test
	DeepAE_model = '/home/ray/Documents/github_code/backward_facing_step_3d/models/saved_models/DeepAE_vel.h5'
	deepAE = load_model(DeepAE_model, compile=False)
	deepae_outputs = deepAE.predict(vol)
	print('The shape of \'DeepAE outputs\' is ',deepae_outputs.shape)
	destinationFile = '/home/ray/Documents/github_code/backward_facing_step_3d/data/DeepAE_outputs'
	transform_vector(deepae_outputs, deepae_outputs.shape[0], originalFile, destinationFile)

	# CAE
	cae = CAE()
	cae_vol, cae_train, cae_test, scaler_u, scaler_v, scaler_w = cae.process_data_for_CAE(path)
	cae.build_CAE_model(encoding_dim = 9)
	cae.train_CAE_model(cae_train, cae_test, epochs = 15, batch_size = 12)
	# test
	cae_model = '/home/ray/Documents/github_code/backward_facing_step_3d/models/saved_models/CAE_vel.h5'
	cae = load_model(cae_model, compile=False)
	cae_outputs = cae.predict(cae_vol)
	print('The shape of \'CAE scaler_outputs\' is ',cae_outputs.shape)
	outputs = cae_outputs.reshape(cae_outputs.shape[0], cae_outputs.shape[1], cae_outputs.shape[2])
	outputs_u = scaler_u.inverse_transform(outputs[:,:,0])
	outputs_v = scaler_v.inverse_transform(outputs[:,:,1])
	outputs_w = scaler_w.inverse_transform(outputs[:,:,2])
	Velocity_DIM = np.dstack((outputs_u, outputs_v, outputs_w))
	print('The shape of \'Velocity\' is ',Velocity_DIM.shape)
	destinationFile = '/home/ray/Documents/github_code/backward_facing_step_3d/data/CAE_outputs'
	transform_vector(Velocity_DIM, Velocity_DIM.shape[0], originalFile, destinationFile)

	# encoder_model = '/home/ray/Documents/github_code/backward_facing_step_3d/models/saved_models/vel_encoder.h5'
	# encoder = load_model(encoder_model, compile=False)
	# code = encoder.predict(train)
	# print('The shape of \'Code\' is ',code.shape)

	# decoder_model = '/home/ray/Documents/github_code/backward_facing_step_3d/models/saved_models/vel_decoder.h5'
	# decoder = load_model(decoder_model, compile = False)
	# outputs = decoder.predict(code)
	# print('The shape of \'scaler_outputs\' is ',outputs.shape)

	# outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], outputs.shape[2])
	# outputs_u = scaler_u.inverse_transform(outputs[:,:,0])
	# outputs_v = scaler_v.inverse_transform(outputs[:,:,1])
	# outputs_w = scaler_w.inverse_transform(outputs[:,:,2])

	# Velocity_DIM = np.dstack((outputs_u, outputs_v, outputs_w))
	# print('The shape of \'Velocity\' is ',Velocity_DIM.shape)

	# originalFile = '/home/ray/Documents/github_code/backward_facing_step_3d/data/backward_facing_step_3d'
	# destinationFile = '/home/ray/Documents/github_code/backward_facing_step_3d/data/outputs'
	# transform(Velocity_DIM, Velocity_DIM.shape[0], originalFile, destinationFile)

	AE_encoder_model = '/home/ray/Documents/github_code/backward_facing_step_3d/models/saved_models/AE_vel_encoder.h5'
	AE_encoder = load_model(AE_encoder_model, compile=False)
	AEcode = AE_encoder.predict(vol)
	print('The shape of \'AE_Code for Predict\' is ',AEcode.shape)
	np.save('AE_Code_for_Predict.npy',AEcode)

	DeepAE_encoder_model = '/home/ray/Documents/github_code/backward_facing_step_3d/models/saved_models/DeepAE_vel_encoder.h5'
	DeepAE_encoder = load_model(DeepAE_encoder_model, compile=False)
	DeepAEcode = DeepAE_encoder.predict(vol)
	print('The shape of \'DeepAE_Code for Predict\' is ',DeepAEcode.shape)
	np.save('Deep_Code_for_Predict.npy',DeepAEcode)

	CAE_encoder_model = '/home/ray/Documents/github_code/backward_facing_step_3d/models/saved_models/CAE_vel_encoder.h5'
	CAE_encoder = load_model(CAE_encoder_model, compile=False)
	CAEcode = CAE_encoder.predict(cae_vol)
	print('The shape of \'AE_Code for Predict\' is ',CAEcode.shape)
	np.save('CAE_Code_for_Predict.npy',CAEcode)