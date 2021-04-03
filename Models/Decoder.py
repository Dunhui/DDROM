from keras.models import Model, load_model
from Load_Data import *
from Model_Processing import *
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

def trans_decoder(tr_outputs, decoder_file_name, di, models_folder, decoder_outputs_name, 
	ori_path, destination_folder, trans_field_name, file_name):

	trans_outputs = np.load(tr_outputs)
	decoder = load_model(decoder_file_name, compile=False)
	decoder_outputs = decoder.predict(trans_outputs)
	data = Load_Data()
	outputs = data.scaler_inverse(di, decoder_outputs, models_folder)
	np.save(decoder_outputs_name,outputs)
	print('The shape of \'decoder outputs\' is ',outputs.shape)

	transform_vector(outputs, outputs.shape[0], ori_path, destination_folder,
	file_name, trans_field_name)