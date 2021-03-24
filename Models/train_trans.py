import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import os ,datetime
from sklearn.preprocessing import MinMaxScaler

def predict_sequences_multiple(model, data, sequence_length, predict_num):

    # data_origin = data[data.shape[0] - sequence_length + 1:,:]
    data = data.reshape(1,data.shape[0],data.shape[1]) #1,20,9
    # print(data.shape)

    print('[Model] Predicting Sequences Multiple...')
    for i in range(predict_num):
      
      # print(i)
      list_p = data[:,:,:] if i == 0 else data[:,i:,:]
      # print(list_p.shape)
      code = model.predict(list_p)
      # print(code.shape)
      code = code.reshape(1,1,code.shape[1])
      # print(code.shape)
      data = np.concatenate((data,code), axis = 1) 
      # print(data.shape)
      # data = data.reshape(data.shape[1],data.shape[2])
      # print(data.shape)

    return data
# def train_transformer(path, Trans_code_name, seq_len, d_k, d_v, n_heads, ff_dim, batch_size, trans_epochs, modelsFolder, transformer_model, transformer_outputs_name):
batch_size = 32
seq_len = 128

d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256
outputs = np.load('Transformer_code_8.npy')
print(outputs.shape)
# outputs =outputs[45:,:]
# transformer_model = './trans_model.h5'



scaler = MinMaxScaler() # data normalization
outputs = scaler.fit_transform(outputs)

test_rate = 0.1
test_point = int(outputs.shape[0] * (1 - test_rate))
train = outputs[:test_point,...]
test = outputs[test_point:,...]
train = np.array(train)
test = np.array(test)


# Training data
X_train, y_train = [], []
for i in range(seq_len, len(train)):
  X_train.append(train[i-seq_len:i]) # Chunks of training data with a length of 128 df-rows
  y_train.append(train[i]) #Value of 4th column (Close Price) of df-row 128+1
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape, y_train.shape)

###############################################################################

# Validation data
X_val, y_val = [], []
for i in range(seq_len, len(test)):
    X_val.append(test[i-seq_len:i])
    y_val.append(test[i])
X_val, y_val = np.array(X_val), np.array(y_val)
print(X_val.shape, y_val.shape)
# ###############################################################################


import transformer
model = transformer.create_model()

callback = tf.keras.callbacks.ModelCheckpoint('transformer_model.h5', 
                                              monitor='loss', 
                                              save_best_only=True, 
                                              verbose=1)

history = model.fit(X_train, y_train, 
                    batch_size=batch_size, 
                    epochs=2, 
                    callbacks=[callback],
                    validation_data=(X_val, y_val))  

ori_data = outputs[0:seq_len,:]
print(ori_data.shape)

predict_num = outputs.shape[0]-seq_len
print(predict_num)

predict_transformer = predict_sequences_multiple(model, ori_data, seq_len, predict_num)
print(predict_transformer.shape)
data = predict_transformer.reshape(predict_transformer.shape[1],predict_transformer.shape[2])
print(data.shape)

# begin_data = data[:seq_len,:]
# data = np.concatenate((begin_data,data), axis = 0) 
# print(data.shape)

# scaler = joblib.load(modelsFolder + '/scaler_code.pkl')


# output_data = scaler.inverse_transform(data)
# print('outputs for transformer', output_data.shape)
# transformer_outputs = path + '/' + transformer_outputs_name
# np.save(transformer_outputs, output_data)