'''
High Performance Extreme Learning Machines (hpelm) application for glass property (Fragility) regression/prediction.
ELM code example - first version with elm model build coding

Created on Tue Jan 22 13:24:00 2019
Modified on: Sun Jan 27 11:36:28 2019
@author: ruhildongol

Cite this open access paper, "High Performance Extreme Learning Machines: A Complete Toolbox for Big Data Applications" in IEEE Access
http://ieeexplore.ieee.org/xpl/articleDetails.jsp?
arnumber=7140733&newsearch=true&queryText=High%20Performance%20Extreme%20Learning%20Machines
@ARTICLE{7140733,
author={Akusok, A. and Bj\"{o}rk, K.-M. and Miche, Y. and Lendasse, A.},
journal={Access, IEEE},
title={High-Performance Extreme Learning Machines: A Complete Toolbox for Big Data Applications},
year={2015},
volume={3},
pages={1011-1025},
doi={10.1109/ACCESS.2015.2450498},
ISSN={2169-3536},
month={},}
'''

import sys
from hpelm import ELM
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.plotly as py 
import plotly.graph_objs as go
from scipy import stats

def process_data(argv):
	'''
	Processes raw excel data by following steps: replacing NaN values, concatenation, filling in 'zero' values, data merger, data shuffle, and normalization

	Arguments:
	argv -- system inputs

	Returns:
	XX -- randomized and normalized training X-values, numpy array of shape (# compositions, # molar%)
	TT -- randomized and normalized training Y-values, numpy array of shape (Fragility value, 1)
	XX_test -- randomized and normalized testing X-values, numpy array of shape (# compositions, # molar%)
	TT_test -- randomized and normalized testing Y-values, numpy array of shape (Fragility value, 1)
	'''
	_, file1 = argv

	# Read excel datasheets
	data_eglass = pd.read_excel('{}'.format(file1), sheet_name='EGlass')
	data_pyrex = pd.read_excel('{}'.format(file1), sheet_name='Pyrex')
	data_TV_panel = pd.read_excel('{}'.format(file1), sheet_name='TV Panel')

	# Replace NaN 'Fragility' values in the data set with mean value 
	data_eglass['Fragility'].fillna(data_eglass['Fragility'].mean(), inplace=True)
	data_pyrex['Fragility'].fillna(data_pyrex['Fragility'].mean(), inplace=True)
	data_TV_panel['Fragility'].fillna(data_TV_panel['Fragility'].mean(), inplace=True)

	# merge all data in one dataframe 
	data_merged = pd.concat([data_TV_panel, data_eglass, data_pyrex], axis=0, ignore_index=True, sort=False)

	# fill NaN values with 'zeros'
	data_merged.fillna(0, inplace=True)

	# move 'Fragility'column to the end
	cols = list(data_merged.columns.values)
	cols.pop(cols.index('Fragility'))
	data_merged = data_merged[cols + ['Fragility']]

	pd.set_option('display.max_rows', 100)
	print(data_merged)											# print merged dataset
	print(data_merged.shape)
	print(data_merged.columns)
	print(data_merged.index)

	# Shuffle rows in pandas dataset
	data_shuffle = data_merged.reindex(np.random.permutation(data_merged.index))			# randomly shuffle data row-wise 
	data_shuffle = data_shuffle.reset_index(drop=True)										# reset index of randomized dataset

	# save merged data to csv 
	data_merged.to_csv('data_merged.csv', header=True, sep=',')					# save data_merged to one big csv file
	data_shuffle.to_csv('data_shuffle.csv', header=True, sep=',')				# save data_shuffle to one big csv file

	# Partition X and Y train and test data sets
	X = data_shuffle.iloc[0:65, 0:20]				# X: first 55 rows for training, columns contain all the composition
	T = data_shuffle.iloc[0:65, 20]					# T: first 55 rows for training, only a single column that contains the property value - Fragility
	X_test = data_shuffle.iloc[65:, 0:20]			# X_test: last 20 rows for testing, columns contain all the composition
	T_test = data_shuffle.iloc[65:, 20]				# T_test: last 20 rows for testing, only a single column that contains the property value - Fragility

	# This 'save' and 'load' steps can be removed 
	# Save the train-test X and Y to csv file
	X.to_csv('X_train.csv', header=False, sep=',')
	T.to_csv('T_train.csv', header=False, sep=',')
	X_test.to_csv('X_test.csv', header=False, sep=',')
	T_test.to_csv('T_test.csv', header=False, sep=',')

	# Load csv 
	X = np.loadtxt('X_train.csv', delimiter=',', usecols=range(1,21))
	T = np.loadtxt('T_train.csv', delimiter=',', usecols=(1))
	X_test = np.loadtxt('X_test.csv', delimiter=',', usecols=range(1,21))
	T_test = np.loadtxt('T_test.csv', delimiter=',', usecols=(1))

	# Normalized to zero mean and unit variance
	XX = (X - X.mean(0)/X.std())
	TT = (T - T.mean(0)/T.std())
	XX_test = (X_test - X_test.mean(0)/X_test.std())
	TT_test = (T_test - T_test.mean(0)/T_test.std())

	# Sanity check - shape assertion
	dim1 = TT.shape[0]
	dim2 = TT_test.shape[0]
	TT = np.reshape(TT, (dim1, 1))
	TT_test = np.reshape(TT_test, (dim2, 1))
	print('-'*50)
	print(dim1, dim2)
	assert(TT.shape == (dim1, 1))	
	assert(TT_test.shape == (dim2, 1))
	
	return XX, TT, XX_test, TT_test

def make_h5_file(data_merged):
	'''
	Converts csv file to hdf5 file

	Argument:
	data_merged -- csv file

	Returns:
	None
	'''

	# Singular file
	ELM.make_hdf5(data_merged, 'data_merged.h5', delimiter=',')

	# Multiple file
	ELM.make_hdf5('x.csv', 'x.h5', delimiter=',')
	ELM.make_hdf5('t.csv', 't.h5', delimiter=',')
	ELM.make_hdf5('xtest.csv', 'xtest.h5', delimiter=',')
	ELM.make_hdf5('ttest.csv', 'ttest.h5', delimiter=',')

	return None

def run_h5_file():
	'''
	Runs hpelm for *.h5 files

	Arguments:

	Returns:

	'''
	# Build model using HPELM
	model = hpelm.HPELM(20,1)

	# Add neurons
	model.add_neurons(100, 'sigm')
	model.add_neurons(9, 'lin')

	# Train hpelm model
	model.train('x.h5', 't.h5')

	# Training error
	model.predict('x.h5', 'y.h5')
	print(model.error('y.h5', 't.h5'))

	# Test error
	model.predict('xtest.h5', 'ytest.h5')
	print(model.error('ytest.h5', 'ttest.h5'))

	return None

def my_plot(Y_true, Y_pred):
	'''
	Plots true Y-values vs predicted Y-value with a linear fit 

	Arguments:
	Y_true -- True measured Y values
	Y_pred -- Predicted Y values
	'''

	# plot show
	slope, intercept, r_value, p_value, std_err = stats.linregress(Y_true[:,0], Y_pred[:,0])
	print('slope: {} and intercept: {}'.format(slope, intercept))
	line = slope*Y_true + intercept
	line2 = Y_true

	# Plot characteristics
	plt.title('True vs Predicted - Test Set')
	plt.plot(Y_true, Y_pred, 'ko', label='values')
	plt.plot(Y_true, line, label='linear fit')
	plt.plot(Y_true, line2, 'b--', label='Y=X')
	plt.legend() #color='black', label='predicted and true values')
	plt.xlabel('True Output')
	plt.ylabel('Predicted Output')
	plt.text(16, 25, 'Y = ' + str(np.round(slope, decimals=2)) + 'X + ' + str(np.round(intercept, decimals=2)) , horizontalalignment='center', verticalalignment='center', fontsize=10)
	plt.show()

	return None

def model_elm(XX, TT, XX_test, TT_test, model_type):
	'''
	Builda elm model using hpelm package

	Arguments:
	XX -- randomized and normalized training X-values, numpy array of shape (# compositions, # molar%)
	TT -- randomized and normalized training Y-values, numpy array of shape (Fragility value, 1)
	XX_test -- randomized and normalized testing X-values, numpy array of shape (# compositions, # molar%)
	TT_test -- randomized and normalized testing Y-values, numpy array of shape (Fragility value, 1)

	Returns:
	model -- save model in ELM format
	'''

	# Hyperparameters
	k = 5						# Use this if model_type == CV
	np.random.seed(10)

	# Build hpelm model
	# ELM(inputs, outputs, classification='', w=None, batch=1000, accelerator=None, precision='double', norm=None, tprint=5)
	model = ELM(20, 1, tprint=5)

	# Add neurons
	model.add_neurons(7, 'tanh')			# Number of neurons with tanh activation
	model.add_neurons(7, 'lin')				# Number of neurons with linear activation

	# if then condition for types of training
	if (model_type == 'CV'):
		print('-'*10 + 'Training with Cross-Validation' + '-'*10)
		model.train(XX, TT, 'CV', k =k)								# Train the model with cross-validation
	elif (model_type == 'LOO'):
		print('-'*10 + 'Training with Leave-One-Out' + '-'*10)
		model.train(XX, TT, 'LOO')									# Train the model with Leave-One-Out
	else:
		print('-'*10 + 'Training with regression' + '-'*10)
		model.train(XX, TT, 'r')									# Train the model with regression

	# Train ELM models
	TTH = model.predict(XX)											# Calculate training error
	YY_test = model.predict(XX_test)								# Calculate testing error	
	print('Model Training Error: ', model.error(TT, TTH))			# Print training error
	print('Model Test Error: ', model.error(YY_test, TT_test))		# Print testing error
	print(str(model))												# Print model information
	print('-'*50)													

	# Call plot function
	my_plot(TT_test, YY_test)

	return model

def main(argv):
	'''
	Implements the main python program

	Arguments:
	argv -- arguments passed during program execution in command line
	'''

	# Get merged data after data processing
	XX, TT, XX_test, TT_test = process_data(argv)

	# ELM model
	# model_type takes values ['None', 'CV', 'LOO']
	model = model_elm(XX, TT, XX_test, TT_test, model_type=None)

	# Save model
	model.save('ELM_model')

# This is main function that run when executed on the command line terminal
if __name__ == '__main__':
	main(sys.argv)

