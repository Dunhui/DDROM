import os
import sys
sys.path.append("/home/ray/.virtualenvs/venv_p3/lib/python3.6/site-packages")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import vtktools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, Flatten, Reshape
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler


def save_model(model, model_name, save_dir):
# function for saving model
	
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	model_path = os.path.join(save_dir, model_name)
	model.save(model_path)

def draw_Acc_Loss(history):
# draw the plot for loss and acc
	plt.figure(1)
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()




