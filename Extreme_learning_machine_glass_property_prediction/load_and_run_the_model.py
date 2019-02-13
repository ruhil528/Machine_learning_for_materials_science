# load and run the saved model

# input new data point of new glasses
# use the built model to predict the property (Y-predicted)

import sys
from hpelm import ELM
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import stats

def process_data(file2):
	'''
	Processes raw excel data by following steps: replacing NaN values, concatenation, filling in 'zero' values, data merger, data shuffle, and normalization

	Arguments:
	file2 -- excel sheet, where row represents different glasses and columns represents oxides
	'''
	# Process excel file containing new data 
	new_data = pd.read_excel('{}'.format(file2))

	# Fill NaN values with zeros
	new_data.fillna(0, inplace=True)

	print(new_data)
	print(new_data.describe)

	X = np.array(new_data)
	print(X)

	return X

def predict_new_data(argv):
	'''
	Implements output prediction for new data

	Arguments:
	argv -- system inputs

	Returns:
	Y -- predicted Y value

	'''

	# file1 = saved model, file2 = excel file with new data
	print(argv)
	_, file1, file2 = argv
	print(file1)
	print(file2)

	# Process the excel data 
	X = process_data(file2)

	# Load model
	model = ELM(20, 1, tprint=5)
	model.load('{}'.format(file1))

	# Predict Y
	Y_predicted = model.predict(X)

	return Y_predicted	

def main(argv):
	'''
	Implements the main python program

	Arguments:
	argv -- arguments passed during program execution in command line
	'''

	# Run this function to predict output for new data 
	Y = predict_new_data(argv)
	print('-'*10 + 'Predicted property for new data!' + '-'*10)
	print(Y)

if __name__ =='__main__':
	main(sys.argv)

