import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import sys
sys.path.append("..")
from Models.AE_model import *
from Models.Transformer import *
from Models.Decoder import *
from Models.pca import *

if __name__=="__main__":  
	
	# data
	ori_path = './flow_past_cylinder_Full Model'# path of full model vtu files
	models_folder = './DOCS' # path of stored models, npys and pkls.
	file_name = '/circle-2d-drag_'# name for each vtu file
	field_name = 'Velocity'# name of selected field
	di = 2 # u,v,w for x,y,z, should be 3, but w = 0 in this example
	data_file_name = models_folder + '/' +field_name + '.npy' #full model data

	data = Load_Data()
	ori_data = data.get_data(ori_path, file_name, field_name, di, data_file_name, models_folder)

	# # AE
	# ae_validation_rate = 0.1 # validation set
	# ae_test_rate = 0.1 # test set 
	ae_encoding_dim = 6 # code dimensions quantity 
	# ae_epochs = 100 # epoch number for AE training part
	# ae_batch_size = 64 # Batch size for AE training part
	# encoder_file_name = 'AE_' + field_name + '_encoder_dim' + str(ae_encoding_dim) + '.h5'# name of encoder model
	# decoder_file_name = 'AE_' + field_name + '_decoder_dim' + str(ae_encoding_dim) + '.h5'# name of decoder model
	# AE_file_name = 'AE_' + field_name + '_dim' + str(ae_encoding_dim) + '.h5' # name of AE whole model

	destination_folder = './ROM' # path of ROM model vtu files
	# new_field_name = field_name + '_AE_dim' + str(ae_encoding_dim) # name of new field restored by AE model
	# Trans_code_name = models_folder + '/' + 'code_dim' + str(ae_encoding_dim) + '.npy' # code compressed by AE

	# AE training 
	# AE(ori_path, file_name, field_name, di, data_file_name, # full model data
	# 	ae_validation_rate, ae_test_rate, ae_encoding_dim,ae_epochs, ae_batch_size, # AE training
	# 	models_folder, encoder_file_name, decoder_file_name, AE_file_name, # store models
	# 	destination_folder, new_field_name, Trans_code_name)# restore data and print codes with definte dimensions.
	

	# PCA
	PCA_name = models_folder + '/' + 'PCA_' + field_name + '_dim' + str(ae_encoding_dim) + '.m' # name of AE whole model
	PCA_code_name = models_folder + '/' + 'PCA_code_dim' + str(ae_encoding_dim) + '.npy' # code compressed by AE
	# PCA_encoder(ori_data, ae_encoding_dim, PCA_name, PCA_code_name)


	tr_validation_rate = 0.1 # validation rate outside the training
	tr_test_rate = 0.1 # test set
	tr_batch_size = 64 # Batch size of transformer training
	tr_epochs = 50 # epoch number of transformer training
	seq_len = 32 # sequence length of transformer training
	d_k = 256 # output number of D_k for query and key
	d_v = 256 # output number of D_v for value
	n_heads = 12 # number of heads
	ff_dim = 256 # dimension of outputs
	start_point = 0 # start time

	tr_model_name = models_folder + '/Transformer_PCA_dim'+ str(ae_encoding_dim) + '.h5' # transformer model name
	tr_outputs = models_folder + '/Transformer_PCA_dim'+ str(ae_encoding_dim) + '_outputs.npy'# transformer outputs

	# Transformer training
	Transformer(tr_validation_rate, tr_test_rate, tr_batch_size, seq_len, tr_epochs, # Transformer training
		d_k, d_v, n_heads, ff_dim, ae_encoding_dim, # Attention
		models_folder, PCA_code_name, tr_model_name, start_point, tr_outputs) # Forecasting

	new_field_name = field_name + '_PCA_dim' + str(ae_encoding_dim)
# # 	trans_field_name = field_name + '_predicted_dim' + str(ae_encoding_dim) # name of predicted field 
# # 	decoder_outputs_name = models_folder + '/' + 'decoder_outputs_dim' + str(ae_encoding_dim) + '.npy' # predicted outputs
	# tr_outputs = PCA_code_name
	# PCA Decoder
	PCA_decoder_name = models_folder + '/' + 'PCA_decoder_outputs_dim' + str(ae_encoding_dim) +'.npy' # predicted outputs
	# PCA_decoder(tr_outputs, PCA_name, PCA_decoder_name)



	PCA_outputs_name = models_folder + '/' + 'PCA_outputs_dim' + str(ae_encoding_dim) +'.npy' # predicted outputs
	# scl_inv(data_file_name, PCA_decoder_name, di, models_folder, PCA_outputs_name)
	
# 	# ae_tf_outputs = models_folder + '/' + 'decoder_outputs_dim6.npy' # predicted outputs
# 	pca_tf_outpits = models_folder + '/' + 'PCA_decoder_outputs_dim6.npy' # predicted outputs
	decoder_outputs_name_2 = models_folder + '/' + 'decoder_outputs_dim' + str(2) + '.npy' # predicted outputs
	decoder_outputs_name_6 = models_folder + '/' + 'decoder_outputs_dim' + str(6) + '.npy' # predicted outputs
	decoder_outputs_name_8 = models_folder + '/' + 'decoder_outputs_dim' + str(8) + '.npy' # predicted outputs
	data_2_name = models_folder + '/' + field_name + '_magnitude_dim_2.npy' # predicted outputs
	data_6_name = models_folder + '/' + field_name + '_magnitude_dim_6.npy' # predicted outputs
	data_8_name = models_folder + '/' + field_name + '_magnitude_dim_8.npy' # predicted outputs

	full_mag_name = models_folder + '/' +'Velocity_magnitude.npy'
	pca_mag_name = models_folder + '/' + field_name + '_magnitude_PCA_dim' + str(ae_encoding_dim) +'.npy' # predicted outputs
	full = np.load(data_file_name) # load original data
	full_mag = magnitude(full)
	np.save(full_mag_name, full_mag)
	pca_data = np.load(PCA_outputs_name)
	print(pca_data.shape)
	pca_mag = magnitude(pca_data)

	data_2 = np.load(data_2_name)
	data_6 = np.load(data_6_name) 
	data_8 = np.load(data_8_name) 
	print(full_mag.shape, pca_mag.shape, data_2.shape, data_6.shape, data_8.shape)
	# cc(full_mag, pca_mag, data_2, data_6, data_8)
	pointNo = 8078 # point of plot figures
	# rmse_over_time(full_mag, pca_mag, data_2, data_6, data_8)
	point_over_time(full_mag, pca_mag, data_2, data_6, data_8, pointNo, field_name)
	# print(full_mag.shape, pca_mag.shape)
	# pcc_of_two(full_mag,pca_mag)
	# rmse_of_two(full_mag, pca_mag)

	 

	
	# transform_vector(outputs, outputs.shape[0], ori_path, destination_folder, file_name, new_field_name)
# 
# 	# Decoder
# 	trans_decoder(tr_outputs, decoder_file_name, di, models_folder, decoder_outputs_name, # decoder
# 		ori_path, destination_folder, trans_field_name, file_name) # transform vector

# 	# Plot figures
# 	# print(ori_data.shape) 	
# 	full = np.load(data_file_name) # load original data
# 	pca_tf = np.load(pca_tf_outpits) # load decoder outputs
# 	outputs_2 = np.load(decoder_outputs_name_2) # load decoder outputs
# 	outputs_6 = np.load(decoder_outputs_name_6) # load decoder outputs
# 	outputs_8 = np.load(decoder_outputs_name_8) # load decoder outputs
# 	cc(full, pca_tf, outputs_2, outputs_6, outputs_8) # plot CC
# 	rmse_over_time(full, pca_tf, outputs_2, outputs_6, outputs_8) # plot RMSE
	# point_over_time(full, outputs_6, pca_tf, pointNo, field_name) # plot magnitude of particular point
	# point_over_time(full, outputs_2, outputs_6, outputs_8, pointNo, field_name) # plot magnitude of particular point
