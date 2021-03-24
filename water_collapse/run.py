import sys
sys.path.append("..")
from Models.AE_model import *
# from Models.Load_Data import *
# from Models.Model_Processing import *

if __name__ == '__main__':
	# data
	path = '../../datasets/data_water_collapse/'	
	ori_path = path + 'origin_data'
	fileName = '/water_collapse_'
	field_name = 'Water::MaterialVolumeFraction'
	di = 1
	data_file_name = '../MaterialVol%.npy'

	# AE
	ae_encoding_dim = 8
	ae_epochs = 1000
	ae_batch_size = 64
	modelsFolder = './saved_models'
	encoderFileName = 'AE_encoder_8.h5'
	decoderFileName = 'AE_decoder_8.h5'
	AEFileName = 'AE_8.h5'
	destinationFolder = path +'ROM_outputs'
	newFieldName = 'MaterialVol_dim_8'
	Trans_code_name = 'Transformer_code_8.npy'

	# AE training
	AE(path, ori_path, fileName, field_name, di, data_file_name, ae_encoding_dim, ae_epochs, ae_batch_size, modelsFolder,encoderFileName, decoderFileName, AEFileName, destinationFolder, newFieldName, Trans_code_name)

	seq_len = 128
	d_k = 256
	d_v = 256
	n_heads = 12
	ff_dim = 256
	trans_batch_size = 256
	trans_epochs=300
	transformer_model = 'Trans_model.h5'
	transformer_outputs = 'output_transformer.npy'

	

