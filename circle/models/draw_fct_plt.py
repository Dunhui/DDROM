import numpy as np
import matplotlib.pyplot as plt


data_ori = np.load("original_data.npy")
data_lstm = np.load("output_lstm.npy")
data_selfatten = np.load("output_selfatten.npy")
data_multiatten = np.load("output_multiatten.npy")
data_transformer = np.load("output_transformer.npy")

print(data_lstm.shape, data_selfatten.shape, data_multiatten.shape)
x = list(range(1,data_lstm.shape[1]+1))#print(x)

i = 350

plt.plot(x,data_ori[i,:], color = 'r', label = 'original_data' )
plt.plot(x,data_lstm[i,:], color = 'b', label = 'LSTM' )
plt.plot(x,data_selfatten[i,:], color = 'g', label = 'LSTM+SelfAttention' )
plt.plot(x,data_multiatten[i,:], color = 'c', label = 'LSTM+MultiAttention' )
plt.plot(x,data_transformer[i,:], color = 'y', label = 'Transformer' )
plt.legend()
plt.xlabel('n')
plt.ylabel('predicted core data')
plt.show()