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
from keras.layers import Dense, LSTM, Activation
from sklearn.metrics import mean_squared_error
from keras.utils.np_utils import to_categorical

# Data Loader imports
from dataloader.openface_dataloader import load_OpenFace_features
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

def load_data(openface_dir, ck_dir, data_type='AUs'):
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
    all_action_units = load_OpenFace_features(openface_dir, features=data_type)
    all_emotions = load_CK_emotions(ck_dir)

    l_AUs, l_emotions = dicts2lists(all_action_units, all_emotions)

    return l_AUs, l_emotions

def build_network(layers=[100], input_shape=(10, 64)):
    model = Sequential()
    
    for layer in layers[:-1]:
        model.add(LSTM(layer, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(layers[-1], input_shape=input_shape))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def main():
    # Fix random seed for reproducibility
    np.random.seed(7)
    
    # Load the datasets
    openface_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+parsed")
    ck_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+")
    for data_type in ['AUs', 'AU_activations', '2Dlandmarks']:
        features, labels = load_data(openface_dir, ck_dir, data_type=data_type)
        labels = to_categorical(labels)

        # Split into train and test sets
        train_size = int(len(features) * 0.67)
        test_size = len(features) - train_size
        x_train, x_test = np.array(features[0:train_size]), np.array(features[train_size:len(features)])
        y_train, y_test = np.array(labels[0:train_size]), np.array(labels[train_size:len(labels)])

        # Normalize length with zero-padding
        maxlen = 71 # Maximum frames of a record from the Cohn-Kanade dataset
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float32')
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float32')
    
        for layers in [[100], [50,50], [15,30], [30,15], [100,100], [50,80], [80,50], [20,20,20], [100,100,100], [50,50,50], [20,50,100], [100,50,20]]:
            for epochs in [3, 6, 10]:
                for batch_size in [1, 3, 6]:
                    print("Iteration parameters:", data_type, layers, epochs, batch_size)

                    # Create and fit the LSTM network
                    model = build_network(layers=layers, input_shape=(x_train.shape[1],x_train.shape[2]))
                    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)
    
                    # Final evaluation of the model
                    scores = model.evaluate(x_test, y_test, verbose=0)
                    print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == "__main__":
    main()