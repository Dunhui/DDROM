
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import sys
sys.path.append("..")
from Models.AE_model import *
from Models.Transformer import *
from Models.Decoder import *
from Models.POD import *
from Models.Load_Data import *

if __name__=="__main__":  
	
	# data
	ori_path = './Full Model'# path of full model vtu files
	models_folder = './DOCS' # path of stored models, npys and pkls.
	file_name = '/circle-2d-drag_'# name for each vtu file
	field_name = 'Velocity'# name of selected field
	di = 2 # u,v,w for x,y,z, should be 3, but w = 0 in this example
	data_file_name = models_folder + '/' +field_name + '.npy' #full model data

	data = Load_Data()
	ori_data = data.get_data(ori_path, file_name, field_name, di, data_file_name, models_folder)
	print(ori_data.shape)

	# AE
	ae_validation_rate = 0.1 # validation set
	ae_test_rate = 0.2 # test set 
	ae_encoding_dim = 6 # code dimensions quantity 
	ae_epochs = 300 # epoch number for AE training part
	ae_batch_size = 32 # Batch size for AE training part
	encoder_file_name = 'AE_' + field_name + '_encoder_dim' + str(ae_encoding_dim) + '.h5'# name of encoder model
	decoder_file_name = 'AE_' + field_name + '_decoder_dim' + str(ae_encoding_dim) + '.h5'# name of decoder model
	AE_file_name = 'AE_' + field_name + '_dim' + str(ae_encoding_dim) + '.h5' # name of AE whole model
	AE_scalered_outputs_name = models_folder + '/' + 'AE_' + field_name + '_predicted_dim' + str(ae_encoding_dim) + '.npy'
	Trans_code_name = models_folder + '/' + 'code_dim' + str(ae_encoding_dim) + '.npy' # code compressed by AE

	destination_folder = './DDROM' # path of ROM model vtu files
	ae_new_field_name = field_name + '_AE_dim' + str(ae_encoding_dim) # name of new field restored by AE model
	pointNo = 5169

	tr_validation_rate = 0.1 # validation rate outside the training
	tr_test_rate = 0.2 # test set
	tr_batch_size = 64 # Batch size of transformer training
	tr_epochs = 100 # epoch number of transformer training
	seq_len = 20 # sequence length of transformer training
	d_k = 16 # output number of D_k for query and key
	d_v = 16 # output number of D_v for value
	n_heads = 8 # number of heads
	ff_dim = 64 # dimen sion of outputs
	start_point = 0 # start time

	# AE
	ae_tr_model_name = models_folder + '/Transformer_ae_dim'+ str(ae_encoding_dim) + '.h5' # transformer model name
	ae_tr_outputs = models_folder + '/Transformer_ae_dim'+ str(ae_encoding_dim) + '_outputs.npy'# transformer outputs
	ae_trans_field_name = field_name + '_ae_predicted_dim' + str(ae_encoding_dim) # name of predicted field 
	ae_decoder_outputs_name = models_folder + '/' + 'AE_decoder_outputs_dim' + str(ae_encoding_dim) + '.npy' # predicted outputs
	ae_error_trans_field_name = field_name + '_ae_predicted_error_dim' + str(ae_encoding_dim) # name of predicted field 

	# POD
	POD_name = models_folder + '/' + 'POD_' + field_name + '_dim' + str(ae_encoding_dim) + '.m' # name of POD whole model
	POD_code_name = models_folder + '/' + 'POD_code_dim' + str(ae_encoding_dim) + '.npy' # code compressed by POD
	POD_decoder_name = models_folder + '/' + 'POD_' + field_name + '_predicted_dim' + str(ae_encoding_dim) + '.npy'
	pod_new_field_name = field_name + '_pod_dim' + str(ae_encoding_dim) # name of new field restored by POD model
	POD_tr_model_name = models_folder + '/Transformer_POD_dim'+ str(ae_encoding_dim) + '.h5' # transformer model name
	POD_tr_outputs = models_folder + '/Transformer_POD_dim'+ str(ae_encoding_dim) + '_outputs.npy'# transformer outputs
	POD_trans_field_name = field_name + '_POD_predicted_dim' + str(ae_encoding_dim) # name of predicted field 
	POD_scaled_decoder_outputs_name = models_folder + '/' + 'POD_scaled_decoder_outputs_dim' + str(ae_encoding_dim) +'.npy' # predicted outputs
	POD_decoder_outputs_name = models_folder + '/' + 'POD_decoder_outputs_dim' + str(ae_encoding_dim) +'.npy' # predicted outputs
	POD_error_trans_field_name = field_name + '_pod_predicted_error_dim' + str(ae_encoding_dim) # name of predicted field 

	if sys.argv[1] == 'AE':
	
		# AE training 
		AE(ori_data, ae_test_rate, ae_validation_rate, ae_encoding_dim, ae_epochs, ae_batch_size, 
		models_folder, encoder_file_name, decoder_file_name, AE_file_name, 
		Trans_code_name, AE_scalered_outputs_name)# restore data and print codes with definte dimensions.

		scl_inv(data_file_name, AE_scalered_outputs_name, di, models_folder, 
		ori_path, destination_folder, file_name, field_name, new_field_name, pointNo)

		# Transformer training
		Transformer(tr_validation_rate, tr_test_rate, tr_batch_size, seq_len, tr_epochs, # Transformer training
			d_k, d_v, n_heads, ff_dim, ae_encoding_dim, # Attention
			models_folder, Trans_code_name, ae_tr_model_name, start_point, ae_tr_outputs) # Forecasting

		AE_trans_decoder(data_file_name, ae_tr_outputs, decoder_file_name, di, models_folder, ae_decoder_outputs_name, # decoder
			ori_path, destination_folder, file_name, field_name, ae_trans_field_name, pointNo)
		
		# Error
		error = calculate_error(data_file_name, ae_decoder_outputs_name)
		transform_vector(error, error.shape[0], ori_path, destination_folder, file_name, ae_error_trans_field_name)


	elif sys.argv[1] == 'POD':	


		# POD
		POD_encoder(ori_data, ae_encoding_dim, POD_name, POD_code_name) 
		POD_decoder(POD_code_name, POD_name, POD_decoder_name)
		scl_inv(data_file_name, POD_decoder_name, di, models_folder, 
			ori_path, destination_folder, file_name, field_name, new_field_name, pointNo)

		# Transformer training
		Transformer(tr_validation_rate, tr_test_rate, tr_batch_size, seq_len, tr_epochs, # Transformer training
			d_k, d_v, n_heads, ff_dim, ae_encoding_dim, # Attention
			models_folder, POD_code_name, POD_tr_model_name, start_point, POD_tr_outputs) # Forecasting

		# POD Decoder
		POD_trans_decoder(data_file_name, POD_tr_outputs, POD_name, di, models_folder, POD_scaled_decoder_outputs_name, POD_decoder_outputs_name, # decoder
		ori_path, destination_folder, file_name, field_name, POD_trans_field_name, pointNo)

		# Error
		error = calculate_error(data_file_name, POD_decoder_outputs_name)
		transform_vector(error, error.shape[0], ori_path, destination_folder, file_name, POD_error_trans_field_name)

	elif sys.argv[1] == 'evaluate':

		full_model_data = magnitude(np.load(data_file_name))

		ae_decoder_3_outputs_name = models_folder + '/' + 'AE_decoder_outputs_dim' + str(3) + '.npy' # predicted outputs
		ae_decoder_4_outputs_name = models_folder + '/' + 'AE_decoder_outputs_dim' + str(4) + '.npy' # predicted outputs
		AE_3_data = magnitude(np.load(ae_decoder_3_outputs_name))
		AE_4_data = magnitude(np.load(ae_decoder_4_outputs_name))

		point_over_time(full_model_data, AE_3_data, AE_4_data, AE_6_data, POD_6_data, pointNo, field_name)

		cc(full_model_data, AE_3_data, AE_4_data, AE_6_data, POD_6_data)
		rmse_over_time(full_model_data, AE_3_data, AE_4_data, AE_6_data, POD_6_data)

	else:
		print('Please enter the extract method AE or POD')
