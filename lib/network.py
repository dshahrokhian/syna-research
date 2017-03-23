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
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils.np_utils import to_categorical

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
    labels = to_categorical(labels)
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    #dataset = scaler.fit_transform(dataset)
    print(features)
    # split into train and test sets
    train_size = int(len(features) * 0.67)
    test_size = len(features) - train_size
    x_train, x_test = np.array(features[0:train_size]), np.array(features[train_size:len(features)])
    y_train, y_test = np.array(labels[0:train_size]), np.array(labels[train_size:len(labels)])
    
    look_back = 1
    maxlen = 71 # Maximum frames of a record from the Cohn-Kanade dataset

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    
    print("created datasets: ")
    print(x_train[0])
    # create and fit the LSTM network
    model = Sequential()
    
    model.add(LSTM(4, input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(Dense(8))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=300, batch_size=1, verbose=2)
    
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == "__main__":
    main()