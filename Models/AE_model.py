import sys
from Models.Load_Data import *
from Models.Model_Processing import *
import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import joblib



class AE_model(object):
	"""DeepAutoencoder Model"""
	def __init__(self):
		super(AE_model, self).__init__()

	def build_ShallowAE_model(self, input_dim, encoding_dim):

		# encoder layers
		input_img = Input(shape=(input_dim, ))
		encoded = Dense(encoding_dim * 8, activation='relu')(input_img)
		encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
		encoded = Dense(encoding_dim)(encoded)

		# decoder layers
		decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
		decoded = Dense(encoding_dim * 8, activation='relu')(decoded)
		decoded = Dense(input_dim, activation='sigmoid')(decoded)

		# this model maps an input to its reconstruction
		self.autoencoder = Model(input_img, decoded)
		self.encoder = Model(input_img, encoded)
		encoded_input = Input(shape=(encoding_dim, ))

		decoder_layer1 = self.autoencoder.layers[-1]
		decoder_layer2 = self.autoencoder.layers[-2]
		decoder_layer3 = self.autoencoder.layers[-3]		
		self.decoder = Model(encoded_input, decoder_layer1(decoder_layer2(decoder_layer3(encoded_input))))

		# configure model 
		self.autoencoder.compile(optimizer='adam', loss = 'mse', metrics = ['accuracy'])

		self.autoencoder.summary()


	def build_DeepAE_model(self, input_dim, encoding_dim):

		input_img = Input(shape=(input_dim, ))
		encoded = Dense(encoding_dim * 32, activation='relu')(input_img)
		encoded = Dense(encoding_dim * 16, activation='relu')(encoded)
		encoded = Dense(encoding_dim * 8, activation='relu')(encoded)
		encoded = Dense(encoding_dim * 4, activation='relu')(encoded)
		encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
		encoded = Dense(encoding_dim)(encoded)

		# "decoded" is the lossy reconstruction of the input
		decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
		decoded = Dense(encoding_dim * 4, activation='relu')(decoded)
		decoded = Dense(encoding_dim * 8, activation='relu')(decoded)
		decoded = Dense(encoding_dim * 16, activation='relu')(decoded)
		decoded = Dense(encoding_dim * 32, activation='relu')(decoded)
		decoded = Dense(input_dim, activation='sigmoid')(decoded)


		# this model maps an input to its reconstruction
		self.autoencoder = Model(input_img, decoded)
		self.encoder = Model(input_img, encoded)
		encoded_input = Input(shape=(encoding_dim, ))

		decoder_layer1 = self.autoencoder.layers[-1]
		decoder_layer2 = self.autoencoder.layers[-2]
		decoder_layer3 = self.autoencoder.layers[-3]		
		decoder_layer4 = self.autoencoder.layers[-4]
		decoder_layer5 = self.autoencoder.layers[-5]
		decoder_layer6 = self.autoencoder.layers[-6]

		# # create the decoder model
		self.decoder = Model(encoded_input, decoder_layer1(decoder_layer2(decoder_layer3(decoder_layer4(decoder_layer5(decoder_layer6(encoded_input)))))))

		# configure model 
		self.autoencoder.compile(optimizer='adam', loss = 'mse', metrics = ['accuracy'])

		self.autoencoder.summary()

		
	def train_AE_model(self, train, test, validation_set, epochs, batch_size, models_folder, encoder_file_name, decoder_file_name, AE_file_name):
		
		self.history_record = self.autoencoder.fit(train, train, epochs = epochs, batch_size = batch_size, validation_data=(test, test))
		draw_Acc_Loss(self.history_record)		
		save_model(self.encoder, encoder_file_name, models_folder)
		save_model(self.decoder, decoder_file_name, models_folder)
		save_model(self.autoencoder, AE_file_name, models_folder)

		print(" DeepAE model trained successfully")  

		scores = self.autoencoder.evaluate(validation_set, validation_set, verbose=1)
		print('Test loss:', scores[0], '\nTest accuracy:', scores[1])
	

def AE(ori_path, file_name, field_name, di, data_file_name, 
	ae_validation_rate, ae_test_rate, ae_encoding_dim, ae_epochs, ae_batch_size, 
	models_folder, encoder_file_name, decoder_file_name, AE_file_name,
	destination_folder, new_field_name, Trans_code_name):

	print("Data loading...")
	# load and pre-processing data
	data = Load_Data()
	inputs_scalered = data.get_data(ori_path, file_name, field_name, di, data_file_name, models_folder)
	train_set, validation_set = data.split_dataset(inputs_scalered, ae_validation_rate)

	random_train_set = data.data_shuffle(train_set)
	train, test = data.split_dataset(random_train_set, test_rate = ae_test_rate)

	# train model
	deepAE = AE_model()
	deepAE.build_DeepAE_model(input_dim = train.shape[1], encoding_dim = ae_encoding_dim)
	deepAE.train_AE_model(train, test, validation_set, epochs = ae_epochs, 
										batch_size = ae_batch_size, 
										models_folder = models_folder, 
										encoder_file_name = encoder_file_name, 
										decoder_file_name = decoder_file_name, 
										AE_file_name = AE_file_name)
	print('AE model is trained successfully.')

	# test 
	ae = load_model(models_folder + "/" + AE_file_name, compile=False)
	ae_outputs = ae.predict(inputs_scalered)
	cc(inputs_scalered,ae_outputs)
	
	outputs = data.scaler_inverse(di, ae_outputs, models_folder)
	ori_data = np.load(data_file_name)
	# print(ori_data.shape, outputs.shape)
	cc(ori_data,outputs)

	# restore data
	transform_vector(outputs, outputs.shape[0], ori_path, destination_folder, file_name, new_field_name)

	# save the code for transformer	 
	encoder = load_model(models_folder + "/" + encoder_file_name, compile=False)
	codes = encoder.predict(inputs_scalered)
	np.save(models_folder + '/' + Trans_code_name, codes)
