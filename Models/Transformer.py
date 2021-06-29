import sys
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from Models.Load_Data import *
from Models.Model_Processing import *


class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
      super(Time2Vector, self).__init__()
      self.seq_len = seq_len

    def build(self, input_shape):
      '''Initialize weights and biases with shape (batch, seq_len)'''
      self.weights_linear = self.add_weight(name='weight_linear',
                                  shape=(int(self.seq_len),),
                                  initializer='uniform',
                                  trainable=True)
      
      self.bias_linear = self.add_weight(name='bias_linear',
                                  shape=(int(self.seq_len),),
                                  initializer='uniform',
                                  trainable=True)
      
      self.weights_periodic = self.add_weight(name='weight_periodic',
                                  shape=(int(self.seq_len),),
                                  initializer='uniform',
                                  trainable=True)

      self.bias_periodic = self.add_weight(name='bias_periodic',
                                  shape=(int(self.seq_len),),
                                  initializer='uniform',
                                  trainable=True)

    def call(self, x):
      '''Calculate linear and periodic time features'''
      x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
      time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
      time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
      
      time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
      time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
      return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)
     
    def get_config(self): # Needed for saving and loading model with custom layer
      config = super().get_config().copy()
      config.update({'seq_len': self.seq_len})
      return config

class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
      super(SingleAttention, self).__init__()
      self.d_k = d_k
      self.d_v = d_v

    def build(self, input_shape):
      self.query = Dense(self.d_k, 
                         input_shape=input_shape, 
                         kernel_initializer='glorot_uniform', 
                         bias_initializer='glorot_uniform')
      
      self.key = Dense(self.d_k, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')
      
      self.value = Dense(self.d_v, 
                         input_shape=input_shape, 
                         kernel_initializer='glorot_uniform', 
                         bias_initializer='glorot_uniform')

    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
      q = self.query(inputs[0])
      k = self.key(inputs[1])

      attn_weights = tf.matmul(q, k, transpose_b=True)
      attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
      attn_weights = tf.nn.softmax(attn_weights, axis=-1)
      
      v = self.value(inputs[2])
      attn_out = tf.matmul(attn_weights, v)
      return attn_out    

class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads):
      super(MultiAttention, self).__init__()
      self.d_k = d_k
      self.d_v = d_v
      self.n_heads = n_heads
      self.attn_heads = list()

    def build(self, input_shape):
      for n in range(self.n_heads):
        self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  
      
      # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
      self.linear = Dense(input_shape[0][-1], 
                          input_shape=input_shape, 
                          kernel_initializer='glorot_uniform', 
                          bias_initializer='glorot_uniform')

    def call(self, inputs):
      attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
      concat_attn = tf.concat(attn, axis=-1)
      multi_linear = self.linear(concat_attn)
      return multi_linear   

class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
      super(TransformerEncoder, self).__init__()
      self.d_k = d_k
      self.d_v = d_v
      self.n_heads = n_heads
      self.ff_dim = ff_dim
      self.attn_heads = list()
      self.dropout_rate = dropout

    def build(self, input_shape):
      self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
      self.attn_dropout = Dropout(self.dropout_rate)
      self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

      self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
      # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7 
      self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1) 
      self.ff_dropout = Dropout(self.dropout_rate)
      self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
    
    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
      attn_layer = self.attn_multi(inputs)
      attn_layer = self.attn_dropout(attn_layer)
      attn_layer = self.attn_normalize(inputs[0] + attn_layer)

      ff_layer = self.ff_conv1D_1(attn_layer)
      ff_layer = self.ff_conv1D_2(ff_layer)
      ff_layer = self.ff_dropout(ff_layer)  
      ff_layer = self.ff_normalize(attn_layer + ff_layer)
      return ff_layer 

    def get_config(self): # Needed for saving and loading model with custom layer
      config = super().get_config().copy()
      config.update({'d_k': self.d_k,
                     'd_v': self.d_v,
                     'n_heads': self.n_heads,
                     'ff_dim': self.ff_dim,
                     'attn_heads': self.attn_heads,
                     'dropout_rate': self.dropout_rate})
      return config  

class Transformer_model(object):
    """docstring for Transformer"""
    def __init__(self):
      super(Transformer_model, self).__init__()

    def create_transformer(self, d_k, d_v, n_heads, ff_dim, seq_len, ae_encoding_dim):
        '''Initialize time and transformer layers'''
        time_embedding = Time2Vector(seq_len)
        attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer4 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer5 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
        attn_layer6 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

        '''Construct model'''
        in_seq = Input(shape=(seq_len, ae_encoding_dim))
        x = time_embedding(in_seq)
        x = Concatenate(axis=-1)([in_seq, x])
        x = attn_layer1((x, x, x))
        x = attn_layer2((x, x, x))
        x = attn_layer3((x, x, x))
        x = attn_layer4((x, x, x))
        x = attn_layer5((x, x, x))
        x = attn_layer6((x, x, x))
        x = Flatten()(x)
        # x = GlobalAveragePooling1D(data_format='channels_first')(x)
        x = Dropout(0.08)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.08)(x)
        out = Dense(ae_encoding_dim, activation='linear')(x)

        model = Model(inputs=in_seq, outputs=out)
       
        return model

    def generate_arrays(self, x, y, batch_size):
        while 1:
          for idx in range(int(np.ceil(len(x)/batch_size))):
            x_excerpt = x[idx*batch_size:(idx+1)*batch_size,...]
            y_excerpt = y[idx*batch_size:(idx+1)*batch_size,...]
            yield x_excerpt, y_excerpt  
   
    def train_transformer(self, model, x_train, y_train, x_val, y_val, test_x, text_y, tr_model_name, tr_batch_size, tr_epochs):

        check_model = ModelCheckpoint(tr_model_name, 
                      monitor='val_loss', 
                      save_best_only=True, 
                      verbose=1)
        reduce_LR = ReduceLROnPlateau(monitor='val_loss', 
                      factor=0.1, 
                      patience=5, 
                      verbose=0, 
                      mode='min', 
                      min_delta=1e-6, 
                      cooldown=0, 
                      min_lr=0)
       
        self.history = model.fit(self.generate_arrays(x_train, y_train, tr_batch_size),
                          steps_per_epoch = np.ceil(len(x_train)/tr_batch_size), 
                          epochs=tr_epochs, 
                          callbacks=[check_model, reduce_LR],
                          validation_data=(x_val, y_val))  

        # draw_Acc_Loss(self.history)  

        scores = model.evaluate(self.generate_arrays(test_x, text_y, tr_batch_size), 
                              steps = np.ceil(len(test_x)/tr_batch_size), 
                              verbose=1)
        print('Test loss:', scores[0], '\nTest accuracy:', scores[1])

    def predict_sequences_multiple(self, model, data, sequence_length, predict_num):

        data = data.reshape(1,data.shape[0],data.shape[1]) #1,20,9
        print('[Model] Predicting Sequences Multiple...')
        for i in range(predict_num):
          list_p = data[:,:,:] if i == 0 else data[:,i:,:]
          code = model.predict(list_p)
          code = code.reshape(1,1,code.shape[1])
          data = np.concatenate((data,code), axis = 1) 

        return data

  
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def Transformer(tr_validation_rate, tr_test_rate, tr_batch_size, seq_len, tr_epochs, 
        d_k, d_v, n_heads, ff_dim, ae_encoding_dim,
        models_folder, Trans_code_name, tr_model_name, start_point, tr_outputs):
    
    codes = np.load(Trans_code_name) 
    # diff_codes = np.diff(codes, n = 1, axis = 0)

    data = Load_Data()
    codes = data.scaler_data(0, codes, models_folder, ae_encoding_dim)# print(np.max(scalered_codes),np.min(scalered_codes), np.mean(scalered_codes), np.median(scalered_codes))
    df_codes = pd.DataFrame(codes)
    print(codes.shape, df_codes.describe())

    data_group = data.create_dataset(codes, seq_len) # return(n, look_back, code_number)
    train_set, test = train_test_split(data_group, test_size=tr_test_rate, shuffle = False)
    print(train_set.shape, test.shape)

    test_x,text_y = test[:,:-1],test[:,-1]# test_x, test_y
    
    x,y = train_set[:,:-1],train_set[:,-1]# train_set_x, train_set_y
    x_train,x_val,y_train,y_val = train_test_split(x, y, test_size=tr_validation_rate, random_state=1)

    transformer = Transformer_model()
    model = transformer.create_transformer(d_k, d_v, n_heads, ff_dim, seq_len, ae_encoding_dim)

    model.compile(loss='mae', optimizer='adam', metrics=['accuracy', 'mae'])
    model.summary()    
    transformer.train_transformer(model, x_train, y_train, x_val, y_val, test_x, text_y, tr_model_name, tr_batch_size, tr_epochs)

    # model.load_weights(tr_model_name)
    # model.compile(loss='mae', optimizer='adam', metrics=['accuracy', 'mae'])
    # model.summary()

    print('\nStart to predict from first sequence\n')
    ori_data = codes[start_point:seq_len,:]
    predict_num = codes.shape[0]-start_point-seq_len

    predict_transformer = transformer.predict_sequences_multiple(model, ori_data, seq_len, predict_num)
    predict_outputs = predict_transformer.reshape(predict_transformer.shape[1],predict_transformer.shape[2])


    # outputs = np.append(codes[0:1,:], predict_outputs, axis = 0)
    # outputs = np.cumsum(outputs, axis = 0, dtype = float)

    if start_point != 0:
        begin_data = scalered_codes[:start_point,:]
        predict_outputs = np.concatenate((begin_data,predict_outputs), axis = 0) 
    # print(codes.shape, predict_outputs.shape)
    output_data = data.scaler_inverse(0, predict_outputs, models_folder, ae_encoding_dim)
    # print(codes.shape, predict_outputs.shape)
    # ae_cc(codes, predict_outputs)
    # # ae_cc(output_data, codes)
    # ae_rmse(codes, predict_outputs)
    # # ae_rmse(output_data, codes)
    # print('Predict succeed. The shape of predicted outputs:', predict_outputs.shape)
    np.save(tr_outputs, output_data)

  

