import sys
from Models.Load_Data import *
from Models.Model_Processing import *
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from sklearn.model_selection import train_test_split
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
		def root_mean_squared_error(true, pred):
			return K.sqrt(K.mean(K.square(pred - true))) 

		# configure model 
		self.autoencoder.compile(optimizer='adam', loss = root_mean_squared_error, metrics = ['accuracy'])
		self.autoencoder.summary()

		

	def train_AE_model(self, train, *validation, test, epochs, batch_size, models_folder, encoder_file_name, decoder_file_name, AE_file_name):
		def generate_arrays(x,batch_size):
			while 1:
				for idx in range(int(np.ceil(len(x)/batch_size))):
					x_excerpt = x[idx*batch_size:(idx+1)*batch_size,...]
					yield x_excerpt, x_excerpt

		check_model = ModelCheckpoint(models_folder + '/' + AE_file_name, 
									monitor='val_loss', 
									save_best_only=True, 
									verbose=1)
		reduce_LR = ReduceLROnPlateau(monitor='val_loss', 
									factor=0.1, 
									patience=5, 
									verbose=0, 
									mode='min', 
									min_delta=0.000001, 
									cooldown=0, 
									min_lr=0)
		# self.history_record = self.autoencoder.fit(train, train, 
		# 										epochs = epochs, 
		# 										batch_size = batch_size, 
		# 										callbacks=[check_model, reduce_LR],
		# 										validation_data=(test, test))
		self.history_record = self.autoencoder.fit(generate_arrays(train, batch_size),
															steps_per_epoch = np.ceil(len(train)/batch_size), 
															epochs = epochs, 
															callbacks=[check_model, reduce_LR],
															# validation_split = 0.2)
															validation_data=(validation, validation))	
		draw_Acc_Loss(self.history_record)		
		save_model(self.encoder, encoder_file_name, models_folder)
		save_model(self.decoder, decoder_file_name, models_folder)
		# save_model(self.autoencoder, AE_file_name, models_folder)

		print(" DeepAE model trained successfully")  

		scores = self.autoencoder.evaluate(generate_arrays(test, batch_size),steps = np.ceil(len(test)/batch_size), verbose=1)
		print('Test loss:', scores[0], '\nTest accuracy:', scores[1])

def generate_predict(x,batch_size):
	while 1:
		for idx in range(int(np.ceil(len(x)/batch_size))):
			x_excerpt = x[idx*batch_size:(idx+1)*batch_size,...]
			yield x_excerpt

def AE(ori_data, ae_test_rate, ae_validation_rate, ae_encoding_dim, ae_epochs, ae_batch_size, 
	models_folder, encoder_file_name, decoder_file_name, AE_file_name, 
	AE_scalered_outputs_name, Trans_code_name):

	# print("Data loading...")
	
	train_set, test = train_test_split(ori_data, test_size=ae_test_rate, random_state=1)
	train, validation = train_test_split(train_set, test_size=ae_validation_rate, random_state=1)

	# train model
	deepAE = AE_model()
	deepAE.build_DeepAE_model(input_dim = train_set.shape[1], encoding_dim = ae_encoding_dim)
	deepAE.train_AE_model(train, validation, test = test, 
										epochs = ae_epochs, 
										batch_size = ae_batch_size, 
										models_folder = models_folder, 
										encoder_file_name = encoder_file_name, 
										decoder_file_name = decoder_file_name, 
										AE_file_name = AE_file_name)
	print('AE model is trained successfully.')

	# test 
	ae = load_model(models_folder + "/" + AE_file_name, compile=False)
	ae_outputs = ae.predict(generate_predict(ori_data, ae_batch_size), steps = np.ceil(len(ori_data)/ae_batch_size))
	np.save(AE_scalered_outputs_name, ae_outputs)

	# save the code for transformer	 
	encoder = load_model(models_folder + "/" + encoder_file_name, compile=False)
	codes = encoder.predict(ori_data)
	np.save(Trans_code_name, codes)

