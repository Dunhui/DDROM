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

def path_value():
	path =  "/home/ray/Documents/data/Rui_2002"
	originalFile = "/home/ray/Documents/data/06-25-test/"# original file
	destinationFile = "/home/ray/Documents/data/predict-1" # destination file

	return path, originalFile, destinationFile

def vel_variable_value():

	vertify_rate = 0.2
	my_epochs = 100
	encoding_dim = 32
	sequence_length = 24
	LSTM_rate = 0.7
	LSTM_epochs = 70 
	predict_num = 960

	return vertify_rate, my_epochs, encoding_dim, sequence_length, LSTM_rate, LSTM_epochs, predict_num


def pre_variable_value():

	vertify_rate = 0.3
	my_epochs = 120
	encoding_dim = 3
	sequence_length =30
	LSTM_rate = 0.8
	LSTM_epochs = 50

	return vertify_rate, my_epochs, encoding_dim, sequence_length, LSTM_rate, LSTM_epochs

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
	velocity = np.load('Velocity.npy') if os.path.exists('Velocity.npy') else get_velocity_data(path, vtu_num)
	pressure = np.load('Pressure.npy') if os.path.exists('Pressure.npy') else get_pressure_data(path, vtu_num)

	return velocity, pressure


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
	# encoded = Dense(encoding_dim * 16, activation='relu')(input_img)
	# encoded = Dense(encoding_dim * 8, activation='relu')(encoded)
	# encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
	# encoded = Dense(encoding_dim, activation='relu')(encoded)


	encoded = Dense(encoding_dim * 16, activation='relu', kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(10e-5))(input_img)
	
	encoded = Dense(encoding_dim * 8, activation='relu', kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(10e-5))(encoded)
	
	encoded = Dense(encoding_dim * 2, activation='relu', kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(10e-8))(encoded)
	encoded = Dense(encoding_dim, activation='relu', kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(10e-8))(encoded)
	
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
	
	decoded = Dense(encoding_dim * 8, activation='relu')(decoded)
	
	decoded = Dense(encoding_dim * 16, activation='relu')(decoded)
	decoded = Dense(vel.shape[1], activation='relu')(decoded)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)

	# Separate Encoder model
	encoder = Model(input_img, encoded)
	encoded_input = Input(shape=(encoding_dim, ))

	# retrieve the layers of the autoencoder model
	# decoder_layer1 = autoencoder.layers[-3]
	# decoder_layer2 = autoencoder.layers[-2]
	# decoder_layer3 = autoencoder.layers[-1]
	# decoder_layer4 = autoencoder.layers[-4]
	# decoder_layer5 = autoencoder.layers[-5]
	# decoder_layer6 = autoencoder.layers[-6]

	# # create the decoder model
	# decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(decoder_layer4(decoder_layer5(decoder_layer6(encoded_input)))))))

	# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
	autoencoder.compile(optimizer='adadelta', loss = 'mean_absolute_error', metrics = ['accuracy'])

	# train the model
	x_train = vel
	# reduce_lr = LearningRateScheduler(scheduler)
	history = autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=32, shuffle=True, validation_split=0.2)
	draw_acc_loss(history)

	# save model
	# encoder.save('vel_encoder.h5') 
	# autoencoder.save('vel_ae.h5')
	# decoder.save('vel_decoder.h5')

	print("ae-model train succeed")  

def ae_pressure(dataset,my_epochs, encoding_dim):

	input_img = Input(shape=(dataset.shape[1], ))

	encoded = Dense(encoding_dim * 32, activation='relu')(input_img)
	encoded = Dense(encoding_dim * 8, activation='relu')(encoded)
	encoded = Dense(encoding_dim, activation='relu')(encoded)
	
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(encoding_dim * 8, activation='relu')(encoded)
	decoded = Dense(encoding_dim * 32, activation='relu')(decoded)
	decoded = Dense(dataset.shape[1], activation='tanh')(decoded)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)

	# Separate Encoder model
	encoder = Model(input_img, encoded)
	encoded_input = Input(shape=(encoding_dim, ))

	# retrieve the layers of the autoencoder model
	# decoder_layer0 = autoencoder.layers[-4]
	decoder_layer1 = autoencoder.layers[-3]
	decoder_layer2 = autoencoder.layers[-2]
	decoder_layer3 = autoencoder.layers[-1]

	# create the decoder model
	decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

	# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
	autoencoder.compile(optimizer='adadelta', loss = 'mean_absolute_error', metrics = ['accuracy'])

	# train the model
	x_train = dataset
	# reduce_lr = LearningRateScheduler(scheduler)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
	history = autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[reduce_lr])
	
	draw_acc_loss(history)

	# save model
	encoder.save('pre_encoder.h5') 
	autoencoder.save('pre_ae.h5')
	decoder.save('pre_decoder.h5')

	print("ae-model train succeed")  


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
	history = model.fit(trainX, trainY, epochs=LSTM_epochs, batch_size=128, validation_data = (testX,testY),verbose = 2)
	_, train_mse = model.evaluate(trainX, trainY, verbose=0)
	_, test_mse = model.evaluate(testX, testY, verbose=0)
	print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
	draw_acc_loss(history)
	# save model
	model.save('pre_lstm.h5')
    
	testYPredict = model.predict(testX)
	trainScore = math.sqrt(mean_absolute_error(testY, testYPredict))
	print("LSTM-model train succeed  ", 'Train Score: %.6f MAE' % (trainScore)) 


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


# copy original data
def copyFiles(sourceDir,targetDir):
    if sourceDir.find("exceptionfolder")>0:
        return

    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir,file)
        targetFile = os.path.join(targetDir,file)

        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or (os.path.exists(targetFile) and (os.path.getsize(targetFile) !=os.path.getsize(sourceFile))):
                open(targetFile, "wb").write(open(sourceFile, "rb").read())
                # print(targetFile+" copy succeeded")

        if os.path.isdir(sourceFile):
            copyFiles(sourceFile, targetFile)

# # create new folder
def mkdir(path):
	folder = os.path.exists(path)

	if folder:                   
		print ("---  We already have this folder name  ---")
		shutil.rmtree(path, ignore_errors=True)
		print ("---  We already delete this folder  ---")

	print ("---  create new folder...---")
	os.makedirs(path)
	("---  OK  ---")

def transform(predicted_pre, originalFile, destinationFile):

	mkdir(destinationFile)     
	copyFiles(originalFile,destinationFile)

# 	# replace velocity with new output data 
	for i in range(predicted_pre.shape[0]):
		f_filename=destinationFile + "/circle-2d-drag_" + str(i+1035)+ ".vtu"
		f_file = vtktools.vtu(f_filename) 
		# velocity_uv = predicted_vel[i].reshape((2,int(predicted_vel.shape[1]/2)))
		# w = np.zeros(velocity_uv.shape[1]).reshape((1,velocity_uv.shape[1]))
		# velocity_uvw = np.vstack((velocity_uv,w)).T
		# f_file.AddVectorField("Velocity_dim", velocity_uvw)
		f_file.AddScalarField("Pressure_predict", predicted_pre[i])
		f_file.Write(f_filename)
	
	print('transform succeed')	

def train_velocity_model(velocity):

	vertify_rate, my_epochs, encoding_dim, sequence_length, LSTM_rate, LSTM_epochs, predict_num = vel_variable_value()

	dataset, vertify = train_and_vertify(velocity,vertify_rate) 
	print("training_dataset shape:",dataset.shape, "vertify_dataset shape:", vertify.shape)

	if not os.path.exists('vel_encoder.h5'):
		ae_velocity(dataset, my_epochs, encoding_dim)

	# if not os.path.exists('vel_lstm.h5'):
	# 	train, test = process_code_for_LSTM(dataset, 'vel_encoder.h5', sequence_length, LSTM_rate)
	# 	LSTM_velocity(train, test, LSTM_epochs)

	return dataset, vertify

def predict_velocity(dataset, predict_num):
	print(np.max(dataset),np.min(dataset), np.mean(dataset), np.median(dataset))
	sequence_length = vel_variable_value()[3]

	encoder = load_model('vel_encoder.h5', compile = False) # encoder test data
	code = encoder.predict(dataset)
	np.save('original_code.npy',code)
	print(np.max(code),np.min(code),np.mean(code), np.median(code),code.shape)

	LSTM = load_model('vel_lstm.h5', compile=False)
	outputs = predict_sequences_multiple(LSTM, code, sequence_length, predict_num)
	np.save('predict_code.npy',outputs)
	print('The shape of \'Velocity\' is ',np.max(outputs),np.min(outputs), np.mean(outputs), np.median(outputs),outputs.shape)


	decoder = load_model('vel_decoder.h5', compile = False) # decoder predicted data
	predicted_vel = decoder.predict(outputs)

	print(np.max(predicted_vel),np.min(predicted_vel))

	return predicted_vel
	


def train_pressure_model(pressure):

	vertify_rate, my_epochs, encoding_dim, sequence_length, LSTM_rate, LSTM_epochs= pre_variable_value()

	dataset, vertify = train_and_vertify(pressure, vertify_rate) 
	print("training_dataset shape:",dataset.shape, "vertify_dataset shape:", vertify.shape)

	if not os.path.exists('pre_ae.h5'):
		ae_pressure(dataset, my_epochs, encoding_dim)

	if not os.path.exists('pre_lstm.h5'):
		train, test = process_code_for_LSTM(dataset, 'pre_encoder.h5', sequence_length, LSTM_rate)
		print(np.max(train),np.min(train))
		LSTM_pressure(train, test, LSTM_epochs)

	return dataset, vertify


def predict_pressure(dataset, predict_num):

	print(np.max(dataset),np.min(dataset), np.mean(dataset), np.median(dataset))
	sequence_length = pre_variable_value()[3]

	encoder = load_model('pre_encoder.h5', compile = False) # encoder test data
	code = encoder.predict(dataset)
	print(np.max(code),np.min(code),np.mean(code), np.median(code), code.shape)

	LSTM = load_model('pre_lstm.h5', compile=False)
	outputs = predict_sequences_multiple(LSTM, code, sequence_length, predict_num)
	print(np.max(outputs),np.min(outputs),np.mean(outputs), np.median(outputs), outputs.shape)

	decoder = load_model('pre_decoder.h5', compile = False) # decoder predicted data
	predicted_vel = decoder.predict(outputs)
	print(np.max(predicted_vel),np.min(predicted_vel),np.mean(predicted_vel), np.median(predicted_vel))

	return predicted_vel

if __name__=="__main__":  

	path, originalFile, destinationFile = path_value()

	velocity, pressure = get_data(path)
	# print(np.max(velocity),np.min(velocity),np.mean(velocity), np.median(velocity))	
	scaler = MinMaxScaler()
	scalered_velocity = scaler.fit_transform(velocity)
	# print(np.max(velocity),np.min(velocity),np.mean(velocity), np.median(velocity))
	# pressure = scaler.fit_transform(pressure) 

	train_vel, vertify_vel = train_velocity_model(scalered_velocity)	
	# train_pre, vertify_pre = train_pressure_model(pressure)

	# predict_num = vel_variable_value()[6] 
	# predicted_vel = predict_velocity(train_vel, predict_num)
	# predicted_pre = predict_pressure(train_pre, predict_num)
	# transform(predicted_pre, originalFile, destinationFile)

# inverse outputs from decoder
	outputs = scaler.inverse_transform(outputs) 