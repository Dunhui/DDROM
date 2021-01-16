import os
import sys
sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
import shutil
from keras.layers import Input, Dense, LSTM, merge, Conv1D, Dropout, Bidirectional, Multiply, Softmax, BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer
import matplotlib.pyplot as plt
import  pandas as pd
import  numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
import tensorflow as tf

from Transformer_model import *

from mkdirs_trans import * 
from LoadVolData import *
from modelProcessing import save_model, draw_Acc_Loss

class SelfAttention(Layer):
    """
    My implementation of a self attention layer.

    This class is of type keras.engine.topology.Layer, so it can be added into any keras model easily. You can read more about self attention layers here ().

    Parameters:
    kqLen (int): length of the key and query vectors generated from input
    valLen (int): length of the value vectors generated from the input
    return_sequence (bool): whether to return all timesteps or just the last one, note time steps are not masked so if return_sequence is True, earlier time steps will contain information about future time steps
    dropout (float): value between 0 and 1 to use for dropout in the layer. Dropout is applied to the key, query, and value matricies and also the output of the layer
    bias (bool): whether or not to train an additive bias for the key, query, and value matricies

    Returns:
    Tensor: if return_sequence = True, tensor is of shape (batch size, sequence length, valLen), else it is (batch size, valLen)
    """

    def __init__(self, kqLen, valLen, return_sequence = True, dropout=0, bias = True, cpu=True, **kwargs):
        self.kqLen = kqLen
        self.valLen = valLen
        self.return_sequence = return_sequence
        self.keyDropout = Dropout(dropout)
        self.queryDropout = Dropout(dropout)
        self.valueDropout = Dropout(dropout)
        self.outputDropout = Dropout(dropout)
        self.layerNorm = BatchNormalization()
        self.softmax = Softmax()
        self.bias = bias

        if cpu:
            self.matmul = tf.matmul
        else:
            self.matmul = K.batch_dot

        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.seqLen = input_shape[1]
        self.inputLen = input_shape[2]
        self.scaleNumber = self.inputLen ** .5

        self.keyW=self.add_weight(name="key_weight",shape=(self.inputLen, self.kqLen),initializer="normal")
        self.queryW=self.add_weight(name="query_weight",shape=(self.inputLen, self.kqLen),initializer="normal")
        self.valW=self.add_weight(name="value_weight",shape=(self.inputLen, self.valLen),initializer="normal")

        if self.bias:
            self.keyB = self.add_weight(name="key_bias",shape=(self.seqLen, self.kqLen),initializer="zeros")
            self.queryB = self.add_weight(name="query_bias",shape=(self.seqLen, self.kqLen),initializer="zeros")
            self.valB = self.add_weight(name="value_bias",shape=(self.seqLen, self.valLen),initializer="zeros")
        else:
            self.keyB = K.constant(0)
            self.queryB = K.constant(0)
            self.valB = K.constant(0)

        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        keys = self.matmul(x,self.keyW) 
        keys = keys + self.keyB
        keys = self.keyDropout(keys)

        queries = self.matmul(x, self.queryW) 
        queries = queries + self.queryB
        queries = self.queryDropout(queries)

        values = self.matmul(x, self.valW)
        values = values + self.valB
        values = self.valueDropout(values)

        outputs = self.matmul(queries, K.permute_dimensions(keys, (0,2,1)))
        outputs = outputs / self.scaleNumber
        outputs = self.softmax(outputs)

        w = self.matmul(outputs, values)

        if not self.return_sequence:
            w = w[:,-1,:]

        w = self.layerNorm(w)

        w = self.outputDropout(w)

        return w

    def compute_output_shape(self, input_shape):
        if self.return_sequence:
            return (input_shape[0], input_shape[1], self.valLen)
        else:
            return (input_shape[0], self.valLen)

    def get_config(self): # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'kqLen': self.kqLen,
                       'valLen': self.valLen,
                       'return_sequence': self.return_sequence,
                       'keyDropout': self.keyDropout,
                       'queryDropout': self.queryDropout,
                       'valueDropout': self.valueDropout,
                       'outputDropout': self.outputDropout,
                       'layerNorm': self.layerNorm,
                       'softmax': self.softmax,
                       'bias': self.bias,
                       'matmul': self.matmul,

                       })
        return config 
 
class AddSinusoidalPositionalEncodings(Layer):
    """
    My implementation of a positional encoding to be used for an attention layer.

    This class is of type keras.engine.topology.Layer, so it can be added into any keras model easily. You can read more about positional encodings here ().
    This layer caches input embeddings in RAM, so if the same shape input is given it does not need to recalculate

    Returns:
    Tensor: will be same shape as the input tensor, positional encodings will be added to the input vector
    """

    def __init__(self, **kwargs):
        super(AddSinusoidalPositionalEncodings, self).__init__(**kwargs)
        self.lookupDic = {}

    def build(self, input_shape):
        super(AddSinusoidalPositionalEncodings, self).build(input_shape)

    def getSinusoidalPositionalEncoding(self, t, d=100):
        a = np.zeros(d)
        for i in range(d):

            if i % 2 == 0:
                sinfun = np.sin
                k = i/2
            else:
                sinfun = np.cos
                k = (i-1)/2

            w = 1/(10000**(2*k/d))
            a[i] = sinfun(w * t)
        return a.astype('float32')

    def getSinusoidalPositionalEncodings(self, batchVector):
        s = tuple(batchVector.shape)
        if s in self.lookupDic:
            return self.lookupDic[s]

        try:
            encodings = np.zeros(batchVector.shape)
        except TypeError: #Will happen when initializing
            return batchVector
        for batchIndex in range(batchVector.shape[0]):
            for timeVectorIndex in range(batchVector[batchIndex].shape[0]):
                encodings[batchIndex][timeVectorIndex] += self.getSinusoidalPositionalEncoding(timeVectorIndex, d=int(batchVector[batchIndex][timeVectorIndex].shape[0]))
        self.lookupDic[s] = tf.constant(encodings.astype('float32'))
        return self.lookupDic[s]

    def call(self, x):
        sinPosEncodings = self.getSinusoidalPositionalEncodings(x)
        x = x + sinPosEncodings
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])

class MultiHeadAttention(Layer):
    """
    My implementation of a Multi-Headed SelfAttention layer.

    This class is of type keras.engine.topology.Layer, so it can be added into any keras model easily. You can read more about self attention layers here ().

    Parameters:
    kqLen (int): length of the key and query vectors generated from input
    valLen (int): length of the value vectors generated from the input
    numHeads (int): number of attention heads for the layer to use
    return_sequence (bool): whether to return all timesteps or just the last one, note time steps are not masked so if return_sequence is True, earlier time steps will contain information about future time steps
    dropout (float): value between 0 and 1 to use for dropout in the layer. Dropout is applied to the key, query, and value matricies and also the output of the layer
    bias (bool): whether or not to train an additive bias for the key, query, and value matricies

    Returns:
    Tensor: if return_sequence = True, tensor is of shape (batch size, sequence length, valLen), else it is (batch size, valLen)
    """
    def __init__(self, kqLen, valLen, numHeads, return_sequence = False, dropout=0, bias=True, cpu=True, **kwargs):
        self.kqLen = kqLen
        self.valLen = valLen
        self.return_sequence = return_sequence
        self.numHeads = numHeads
        self.dropout = dropout
        self.bias = bias
        self.cpu = cpu
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.seqLen = input_shape[1]
        self.inputLen = input_shape[2]

        self.attentionLayers = []
        for i in range(self.numHeads):
            self.attentionLayers.append(SelfAttention(kqLen=self.kqLen, valLen=self.valLen, return_sequence=True, dropout=self.dropout, bias=self.bias, cpu=self.cpu))


        self.resizeWeight=self.add_weight(name="resize_weight",shape=(self.numHeads * self.valLen, self.valLen),initializer="normal")
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, x):
        outputs = self.attentionLayers[0](x)

        for layerIndex in range(1,len(self.attentionLayers)):
            outputs = tf.concat([outputs, self.attentionLayers[layerIndex](x)], 2)
        output = tf.matmul(outputs, self.resizeWeight)

        if not self.return_sequence:
            output = output[:,-1,:]
        return output

    def compute_output_shape(self, input_shape):
        if self.return_sequence:
            return (input_shape[0], input_shape[1], self.valLen)
        else:
            return (input_shape[0], self.valLen)

    def get_config(self): # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'kqLen': self.kqLen,
            'valLen': self.valLen,
            'return_sequence': self.return_sequence,
            'numHeads': self.numHeads,
            'dropout': self.dropout,
            'bias': self.bias,
            'cpu': self.cpu,

        })
        return config

class FCT_models(object):
    """docstring for FCT_models"""
    def __init__(self):
        super(FCT_models, self).__init__()
            
            
    def LSTM_model(self,outputDim):

        # model structure
        model = Sequential()
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(outputDim))

        # compile model
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model
        
    def SelfAtten_model(self, seqLen, inputLen, outputDim, learning_rate=.0035):

        input_ = Input(shape=(seqLen,inputLen))
        x = AddSinusoidalPositionalEncodings()(input_)
        x = SelfAttention(120, 120, return_sequence=False, dropout=.2)(x)
        x = Dense(outputDim, activation='softmax')(x)

        model = Model(inputs=input_, outputs=x)
        
        opt = Adam(lr = learning_rate)
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])  

        return model
        
    def MultiAtten_model(self, seqLen, inputLen, outputDim, learning_rate=.0035):

        input_ = Input(shape=(seqLen,inputLen))
        x = AddSinusoidalPositionalEncodings()(input_)
        x = MultiHeadAttention(120, 120, 8, return_sequence=False, dropout=.2)(x) #uncomment to use multihead attention with 8 heads
        x = Dense(outputDim, activation='softmax')(x)

        model = Model(inputs=input_, outputs=x)

        opt = Adam(lr = learning_rate)
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

        return model

    def Transformer(self, seq_len, TIME_STEPS, d_k, d_v, n_heads, ff_dim):

        '''Initialize time and transformer layers'''
        time_embedding = Time2Vector(seq_len)
        attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

        '''Construct model'''
        in_seq = Input(shape=(seq_len, 9))
        x = time_embedding(in_seq)
        x = Concatenate(axis=-1)([in_seq, x])
        x = attn_layer1((x, x, x))
        x = attn_layer2((x, x, x))
        x = attn_layer3((x, x, x))
        x = GlobalAveragePooling1D(data_format='channels_first')(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        out = Dense(9, activation='relu')(x)

        model = Model(inputs=in_seq, outputs=out)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return model

def predict_sequences_multiple(model, data, sequence_length, predict_num):
	
	# data_origin = data[data.shape[0] - sequence_length + 1:,:]
	data = data.reshape(1,data.shape[0],data.shape[1]) #1,20,9

	print('[Model] Predicting Sequences Multiple...')
	for i in range(predict_num+1):
		list_p = data[:,:,:] if i == 0 else data[:,i:,:]
		code = model.predict(list_p)
		code = code.reshape(1,1,code.shape[1])
		data = np.concatenate((data,code), axis = 1) 
	data = data.reshape(data.shape[1],data.shape[2])
	return data

if __name__=="__main__":  

    TIME_STEPS = 20
    
    data_ae = np.load("/home/ray/Documents/github_code/circle/data/AE_Code_for_Predict.npy")
    data_deepae = np.load("/home/ray/Documents/github_code/circle/data/Deep_Code_for_Predict.npy")
    data_cae = np.load("/home/ray/Documents/github_code/circle/data/CAE_Code_for_Predict.npy")
    print(data_ae.shape, data_deepae.shape, data_cae.shape)

    save_dir = os.path.join(os.getcwd(), 'saved_models')

    data = data_deepae # data_ae or data_cae


# def FCT(data = data_deepae):
    scaler = MinMaxScaler()
    data= scaler.fit_transform(data)
    dataloaded = LoadmyData()
    train, test = dataloaded.train_and_test(data, test_rate=0.2) 
    print(train.shape, test.shape)
    train_x, train_y = dataloaded.create_dataset(train, TIME_STEPS)
    test_x, test_y = dataloaded.create_dataset(test, TIME_STEPS)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    outputDim = train_x.shape[2]

    FCT = FCT_models()
    ori_data = test_x[0,:]
    sequence_length = 20
    predict_num = test_x.shape[0]

    # LSTM
    model = FCT.LSTM_model(outputDim)
    history = model.fit(train_x, train_y, epochs=200, batch_size=50, validation_split=0.2)
    draw_Acc_Loss(history)
    # save_model(model, 'vel_LSTM.h5', save_dir)
    output = model.predict(test_x)
    scores = model.evaluate(test_x, test_y, verbose=1)
    trainScore = math.sqrt(mean_absolute_error(test_y, output))
    print('LSTM : Test loss:', scores[0], '  Test accuracy:', scores[1], '  Train Score:', trainScore)

    predict_lstm = predict_sequences_multiple(model, ori_data, sequence_length, predict_num)
    print(predict_lstm.shape)
    np.save('output_lstm.npy',predict_lstm)

    # SelfAttention 
    model_selfAt = FCT.SelfAtten_model(train_x.shape[1], train_x.shape[2], outputDim)
    model_selfAt.summary()
    history = model_selfAt.fit(train_x, train_y, epochs=200, batch_size=50, validation_split=0.2)
    draw_Acc_Loss(history)
    # save_model(model_selfAt, 'vel_selfAtten.h5', save_dir) 
    output = model_selfAt.predict(test_x)                                                   
    scores = model_selfAt.evaluate(test_x, test_y, verbose=1)
    trainScore = math.sqrt(mean_absolute_error(test_y, output))
    print('LSTM + SelfAttention : Test loss:', scores[0], '  Test accuracy:', scores[1], '  Train Score:', trainScore)
    predict_selfAttention = predict_sequences_multiple(model_selfAt, ori_data, sequence_length, predict_num)
    print(predict_selfAttention.shape)
    np.save('output_selfatten.npy',predict_selfAttention)



    # MultiAttention
    model = FCT.MultiAtten_model(train_x.shape[1], train_x.shape[2], outputDim)
    model.summary()
    history = model.fit(train_x, train_y, epochs=200, batch_size=50, validation_split=0.2)
    draw_Acc_Loss(history)
    # save_model(model, 'vel_multiAtten.h5', save_dir)
    output = model.predict(test_x)
    scores = model.evaluate(test_x, test_y, verbose=1)
    trainScore = math.sqrt(mean_absolute_error(test_y, output))
    print('LSTM + MultiAttention : Test loss:', scores[0], '  Test accuracy:', scores[1], '  Train Score:', trainScore)
    predict_multiAttention = predict_sequences_multiple(model, ori_data, sequence_length, predict_num)
    print(predict_multiAttention.shape)
    np.save('output_multiatten.npy',predict_multiAttention)

	# Transformer 
    seq_len = 20
    TIME_STEPS = 150
    d_k = 256
    d_v = 256
    n_heads = 12
    ff_dim = 256

    model = FCT.Transformer(seq_len, TIME_STEPS, d_k, d_v, n_heads, ff_dim)
    model.summary()

    callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding_avg.h5', monitor='val_loss', save_best_only=True,verbose=1)
    history = model.fit(train_x, train_y, batch_size=64, epochs=200, callbacks=[callback],validation_split=0.2)  
    draw_Acc_Loss(history)
    # save_model(model, 'vel_Transformer.h5', save_dir)                                             
    output = model.predict(test_x)
    scores = model.evaluate(test_x, test_y, verbose=1)
    trainScore = math.sqrt(mean_absolute_error(test_y, output))
    print('Transformer : Test loss:', scores[0], '  Test accuracy:', scores[1], '  Train Score:', trainScore)

    predict_transformer = predict_sequences_multiple(model, ori_data, sequence_length, predict_num)
    print(predict_transformer.shape)
    np.save('output_transformer.npy',predict_transformer)



