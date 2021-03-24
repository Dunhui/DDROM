from AE_model import *
from trans_train import *
from trans_decoder import *

if __name__=="__main__":  

	# data
	path = '../../datasets/flow_past_cylinder/'	
	ori_path = path + 'Rui_2002'
	fileName = '/circle-2d-drag_'
	field_name = 'Velocity'
	di = 2
	data_file_name = '../Velocity_U_V.npy'

	# AE
	ae_encoding_dim = 8
	ae_epochs = 40
	ae_batch_size = 64
	modelsFolder = './saved_models'
	encoderFileName = 'AE_vel_encoder_8.h5'
	decoderFileName = 'AE_vel_decoder_8.h5'
	AEFileName = 'AE_vel_8.h5'
	destinationFolder = path +'ROM_outputs'
	newFieldName = 'Velocity_dim_8'
	Trans_code_name = 'Transformer_code_8.npy'

	# AE training
	# AE(path, ori_path, fileName, field_name, di, data_file_name, ae_encoding_dim, ae_epochs, ae_batch_size, modelsFolder,encoderFileName, decoderFileName, AEFileName, destinationFolder, newFieldName, Trans_code_name)

	seq_len = 128
	d_k = 256
	d_v = 256
	n_heads = 12
	ff_dim = 256
	trans_batch_size = 256
	trans_epochs=300
	transformer_model = 'Trans_model.h5'
	transformer_outputs = 'output_transformer.npy'
	# train_transformer(path, Trans_code_name, seq_len, d_k, d_v, n_heads, ff_dim, trans_batch_size, trans_epochs, modelsFolder, transformer_model)

	trans_FieldName = 'fore_Velocity_dim'
	trans_decoder(path, ori_path, destinationFolder, trans_FieldName, fileName, modelsFolder, decoderFileName, transformer_outputs)