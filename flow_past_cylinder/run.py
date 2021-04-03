import sys
sys.path.append("..")
from Models.AE_model import *
from Models.Transformer import *


if __name__=="__main__":  

	# data
	ori_path = './Full Model'# path of full model vtu files
	models_folder = './DOCS' # path of stored models, npys and pkls.
	file_name = '/circle-2d-drag_'# name for each vtu file
	field_name = 'Velocity'# name of selected field
	di = 2 # u,v,w for x,y,z, should be 3, but w = 0 in this example
	data_file_name = models_folder + '/' +field_name + '.npy' #full model data

	# AE
	ae_validation_rate = 0.1 # validation set
	ae_test_rate = 0.2 # test set 
	ae_encoding_dim = 8 # code dimensions quantity 
	ae_epochs = 30 # epoch number for AE training part
	ae_batch_size = 64 # Batch size for AE training part
	encoder_file_name = 'AE_' + field_name + '_encoder_dim' + '.h5'# name of encoder model
	decoder_file_name = 'AE_' + field_name + '_decoder_dim' + str(ae_encoding_dim) + '.h5'# name of decoder model
	AE_file_name = 'AE_' + field_name + '_dim' + str(ae_encoding_dim) + '.h5' # name of AE whole model

	destination_folder = './ROM' # path of ROM model vtu files
	new_field_name = field_name + '_AE_dim' + str(ae_encoding_dim) # name of new field restored by AE model
	Trans_code_name = models_folder + '/' + 'code_dim' + str(ae_encoding_dim) + '.npy' # code compressed by AE

	# AE training 
	AE(ori_path, file_name, field_name, di, data_file_name, # full model data
		ae_validation_rate, ae_test_rate, ae_encoding_dim,ae_epochs, ae_batch_size, # AE training
		models_folder, encoder_file_name, decoder_file_name, AE_file_name, # store models
		destination_folder, new_field_name, Trans_code_name)# restore data and print codes with definte dimensions.
	
	tr_validation_rate = 0.1 # validation rate outside the training
	tr_test_rate = 0.1 # test set
	tr_batch_size = 32 # Batch size of transformer training
	tr_epochs = 30 # epoch number of transformer training
	seq_len = 128 # sequence length of transformer training
	d_k = 256 # output number of D_k for query and key
	d_v = 256 # output number of D_v for value
	n_heads = 12 # number of heads
	ff_dim = 256 # dimension of outputs
	start_point = 0 # start time

	tr_model_name = models_folder + '/Transformer_dim'+ str(ae_encoding_dim) + '.h5' # transformer model name
	tr_outputs = models_folder + '/Transformer_dim'+ str(ae_encoding_dim) + '_outputs.npy'# transformer outputs

	# Transformer training
	train_transformer(tr_validation_rate, tr_test_rate, tr_batch_size, seq_len, tr_epochs, # Transformer training
		d_k, d_v, n_heads, ff_dim, ae_encoding_dim, # Attention
		models_folder, Trans_code_name, tr_model_name, start_point, tr_outputs) # Forecasting

	trans_field_name = field_name + '_predicted_dim' + str(ae_encoding_dim) # name of predicted field 
	decoder_outputs_name = models_folder + '/' + 'decoder_outputs_dim' + str(ae_encoding_dim) + '.npy' # predicted outputs
	pointNo = 1212 # point of plot figures

	# Decoder
	trans_decoder(tr_outputs, decoder_file_name, di, models_folder, decoder_outputs_name, # decoder
		ori_path, destination_folder, trans_field_name, file_name) # transform vector

	# Plot figures
	ori_data = np.load(data_file_name) # load original data
	outputs = np.load(decoder_outputs_name) # load decoder outputs
	cc(ori_data, outputs) # plot CC
	rmse_over_time(ori_data, outputs) # plot RMSE
	point_over_time(ori_data, outputs, pointNo, field_name) # plot magnitude of particular point