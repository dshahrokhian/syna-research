# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - LSTM Network
===============================================================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

# Neural Network imports
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Data Loader imports
from dataloader.openface_dataloader import load_AU_activations
from dataloader.ck_dataloader import load_CK_emotions

def load_data(openface_dir, ck_dir):
	""" 
    Extracts OpenFace Action Units features and CK+ Emotions labels.

    Parameters
    ----------
    openface_dir : root directory of the parsed CK+ dataset
	ck_dir : root directory of the CK+ dataset
    
    Returns
    -------
    Dict, Dict
        {Record identifier : {timestamp : {AU code : AU activation value}}},
		{Record identifier : Emotion identifier}
    """
	all_action_units = load_AU_activations(openface_dir)
	all_emotions = load_CK_emotions(ck_dir)
	
	return all_action_units, all_emotions


def main():
    # fix random seed for reproducibility
	np.random.seed(7)
	
	# load the datasets
	openface_dir = os.path.join(os.getcwd(), "datasets/ck+parsed")
	ck_dir = os.path.join(os.getcwd(), "datasets/ck+")

	all_action_units, all_emotions = load_data(openface_dir, ck_dir)
	action_units = all_action_units["S005_001"]
	print(action_units)

	emotion = all_emotions["S005_001"]
	print(emotion)

	dataframe = pandas.DataFrame.from_dict(action_units, orient='index')
	dataset = dataframe.values
	
	dataset = dataset.astype('float32')
	
	print(dataset[0][2])
	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	#dataset = scaler.fit_transform(dataset)
	print("later")
	print(dataset[0][2])
	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	print("split datasets:")
	print(train[0])
	exit()
	# convert an array of values into a dataset matrix
	def create_dataset(dataset, look_back=1):
		dataX, dataY = [], []
		for i in range(len(dataset)-look_back-1):
			a = dataset[i:(i+look_back), 0]
			dataX.append(a)
			dataY.append(dataset[i + look_back, 0])
		return np.array(dataX), np.array(dataY)
	
	# reshape into X=t and Y=t+1
	look_back = 1
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	print("created datasets: ")
	print(trainX[0], trainY[0])
	
	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	print("reshaped datasets: ")
	print(trainX[0])
	
	# create and fit the LSTM network
	model = Sequential()
	
	model.add(LSTM(4, input_shape=[look_back,1]))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)
	
	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	
	# shift train predictions for plotting
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	# plot baseline and predictions
	plt.plot(scaler.inverse_transform(dataset))
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()

if __name__ == "__main__":
    main()