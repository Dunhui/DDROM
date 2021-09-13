import sys
from Models.Load_Data import *
from Models.Model_Processing_fpc import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

def POD_encoder(x, n, POD_name, POD_code_name):
	'''
	x: input
	n: encoding dim
	POD_name: POD model name
	POD_code_name: POD code outputs name
	
	pod.explained_variance_ratio_ = POD model ratio
	principalComponents: POD code outputs
	'''
	pod = PCA(n_components = n)
	principalComponents = pod.fit_transform(x)
	ratio = pod.explained_variance_ratio_
	print(pod.explained_variance_ratio_)
	POD_ratio(pod.explained_variance_ratio_)
	# barries(ratio, 0.98)
	joblib.dump(pod, POD_name)
	np.save(POD_code_name, principalComponents)

def POD_decoder(x_name, POD_name):
	'''
	# x: predicted code  
	# POD_name: POD model name
	'''
	x = np.load(x_name)
	pod = joblib.load(POD_name)
	outputs = pod.inverse_transform(x)
	# np.save(POD_decoder_outputs_name, outputs)
	return outputs

def barries(data, barry):
	sum = 0
	for i in range(len(data)):
		sum = sum + data[i]
		if sum > barry:
			break
	print(i, sum, i-1, sum-data[i])


