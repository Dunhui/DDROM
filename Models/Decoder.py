from keras.models import Model, load_model
from Models.Load_Data import *
from Models.Model_Processing import *
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

def AE_scl_inv(data_file_name, AE_scalered_outputs_name, di, models_folder, 
	ori_path, destination_folder, file_name, ae_field_name):

	# load ori_data and ae outputs
	data = np.load(data_file_name)
	ae_outputs = np.load(AE_scalered_outputs_name)

	# scaler inverse
	if np.max(data)>1.1 or np.min(data)<-0.1:
		data = Load_Data()
		ae_outputs = data.scaler_inverse(di, ae_outputs, models_folder)

	print('data scaler inversed, the shape of ae outputs is :', ae_outputs.shape)

	# plot 
	ae_cc(data,ae_outputs)
	ae_rmse(data, ae_outputs)

	# transform vector to vtu files
	transform_vector(ae_outputs, ae_outputs.shape[0], ori_path, destination_folder,
	file_name, ae_field_name)


def trans_decoder(tr_outputs, decoder_file_name, di, models_folder, decoder_outputs_name, # decoder
		ori_path, destination_folder, trans_field_name, file_name):
	
	# load ori_data and transformer outputs
	data = np.load(data_file_name)
	trans_outputs = np.load(tr_outputs)

	# Decoder transformer outputs
	decoder = load_model(models_folder + "/" + decoder_file_name, compile=False)
	decoder_outputs = decoder.predict(trans_outputs)
	
	# scaler inverse
	if np.max(data)>1.1 or np.min(data)<-0.1:
		data = Load_Data()
		decoder_outputs = data.scaler_inverse(di, decoder_outputs, models_folder)

	# plot 
	ae_cc(data,decoder_outputs)
	ae_rmse(data, decoder_outputs)

	# save decoder outputs
	np.save(decoder_outputs_name,decoder_outputs)
	print('The shape of \'decoder outputs\' is ',decoder_outputs.shape)
	outputs = np.load(decoder_outputs_name)

	# transform vector to vtu files
	transform_vector(outputs, outputs.shape[0], ori_path, destination_folder,
	file_name, trans_field_name)

def scl_inv(data_file_name, PCA_decoder_name, di, models_folder, PCA_outputs_name):
	# load ori_data and ae outputs
	ori_data = np.load(data_file_name)
	outputs = np.load(PCA_decoder_name)
	print(ori_data.shape, outputs.shape)
	# scaler inverse
	if np.max(ori_data)>1.1 or np.min(ori_data)<-0.1:
		data = Load_Data()
		outputs = data.scaler_inverse(di, outputs, models_folder)

	print('data scaler inversed, the shape of ae outputs is :', outputs.shape)

	# plot 
	pcc_of_two(ori_data,outputs)
	rmse_of_two(ori_data, outputs)

	np.save(PCA_outputs_name, outputs)

	# # transform vector to vtu files
	# transform_vector(outputs, outputs.shape[0], ori_path, destination_folder,
	# file_name, trans_field_name)