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

def dicts2lists(dict_action_units, dict_emotions):
    """ 
    Converts the dictionaries of the dataloaders into lists, containing only
    record with identifiers present in both the Action Units dictionary and 
    Emotions dictionary, and ordered by record identifiers.

    Parameters
    ----------
    dict_action_units : {Record identifier : 
                            {timestamp :    
                                {AU code : AU activation value}
                            }
                        }
    dict_emotions : {Record identifier : Emotion identifier}
    
    Returns
    -------
    List, List
        [records, samples, AUs], [emotion identifiers]
    """
    l_AUs = []
    l_emotions = []

    for record_id, activations in dict_action_units.items():
        if record_id in dict_emotions:
            record_AUs = []
            for timestamp in sorted(activations.keys()):
                record_AUs.append(list(activations[timestamp].values()))
            l_AUs.append(record_AUs)
            l_emotions.append(dict_emotions[record_id])
    
    return l_AUs, l_emotions

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

    l_AUs, l_emotions = dicts2lists(all_action_units, all_emotions)

    return l_AUs, l_emotions

def main():
    # fix random seed for reproducibility
    np.random.seed(7)
    
    # load the datasets
    openface_dir = os.path.join(os.getcwd(), "datasets/ck+parsed")
    ck_dir = os.path.join(os.getcwd(), "datasets/ck+")
    features, labels = load_data(openface_dir, ck_dir)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    #dataset = scaler.fit_transform(dataset)
    print(features)
    # split into train and test sets
    train_size = int(len(features) * 0.67)
    test_size = len(features) - train_size
    train_features, test_features = features[0:train_size], features[train_size:len(features)]
    train_labels, test_labels = labels[0:train_size], labels[train_size:len(labels)]
    print(train_features)
    # convert an array of values into a dataset matrix
    def create_dataset(features, labels, look_back=1):
        dataX, dataY = [], []
        print(features)
        print("yo")
        for i in range (len(features)):
            a = []
            for j in range(len(features[i])-look_back-1):
                b = features[i][j:(j+look_back)]
                a.append(b)
            dataX.append(a)
            dataY.append(labels[i])
        print(dataX)
        return np.array(dataX), np.array(dataY)
    
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train_features, train_labels, look_back)
    testX, testY = create_dataset(test_features, test_labels,  look_back)
    print("created datasets: ")
    print(trainX, trainY)
    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], look_back, trainX.shape[3]))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], look_back, testX.shape[3]))
    print("reshaped datasets: ")
    print(trainX[0])
    
    # create and fit the LSTM network
    model = Sequential()
    
    model.add(LSTM(4, input_shape=(trainX.shape[1],look_back,18)))
    model.add(Dense(1))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
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