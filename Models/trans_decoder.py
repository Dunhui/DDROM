import sys
sys.path.append('/home/ray/Documents/github_code/water_collapse/models')
from keras.models import Model, load_model
import LoadVolData 
from mkdirs_trans import *
import numpy as np



path = '../data/origin_data/'	
fileName = '/water_collapse_'
# modelsFolder = 
# encoderFileName = 
# decoderFileName = 
# AEFileName = 
destinationFolder = '../data/trans_outputs'
newFieldName = 'fore_MaterialVol%_dim'

data = np.load("../data/output_transformer.npy")

print(data.shape)

decoder_model = './saved_models/DeepAE_vel_decoder_8.h5'
decoder = load_model(decoder_model, compile=False)
decoder_outputs = decoder.predict(data)
print('The shape of \'decoder outputs\' is ',decoder_outputs.shape, '\nStart to update data in vtu files...')
transform_vector(decoder_outputs, decoder_outputs.shape[0], path, destinationFolder, fileName, newFieldName)