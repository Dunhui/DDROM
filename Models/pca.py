import sys
from Models.Load_Data import *
from Models.Model_Processing import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

def PCA_encoder(x, n, PCA_name, PCA_code_name):
	'''
	x: input
	n: encoding dim
	PCA_name: PCA model name
	PCA_code_name: PCA code outputs name
	
	pca.explained_variance_ratio_ = PCA model ratio
	principalComponents: PCA code outputs
	'''
	pca = PCA(n_components = n)
	principalComponents = pca.fit_transform(x)
	ratio = pca.explained_variance_ratio_
	# print(ratio.shape, x.shape, principalComponents.shape)
	# barries(ratio, 0.98)
	joblib.dump(pca, PCA_name)
	np.save(PCA_code_name, principalComponents)

def PCA_decoder(x_name, PCA_name, PCA_decoder_name):
	'''
	# x: predicted code  
	# PCA_name: PCA model name
	# PCA_decoder_name: decoder outputs
	'''
	x = np.load(x_name)
	pca = joblib.load(PCA_name)
	outputs = pca.inverse_transform(x)
	print(x.shape, outputs.shape)
	np.save(PCA_decoder_name, outputs)

def barries(data, barry):
	sum = 0
	for i in range(len(data)):
		sum = sum + data[i]
		if sum > barry:
			break
	print(i, sum, i-1, sum-data[i])


