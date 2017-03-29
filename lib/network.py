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
from dataloader.afew_dataloader import load_AFEW_emotions

def dicts2lists(dict_action_units, dict_emotions):
    """ 
    Converts the dictionaries of the dataloaders into lists, containing only
    records with identifiers present in both the Action Units dictionary and 
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
    first = True
    for record_id, activations in dict_action_units.items():
        if record_id in dict_emotions:
            record_AUs = []
            for timestamp in sorted(activations.keys()):
                record_AUs.append(list(activations[timestamp].values()))
            if (first):
                print(record_id)
                first = False
            l_AUs.append(record_AUs)
            l_emotions.append(dict_emotions[record_id])
    
    return l_AUs, l_emotions

def load_data(openface_dir, emotion_dir, dataset, data_type='AUs', set='Train'):
    """ 
    Extracts OpenFace Action Units features and CK+/AFEW Emotion labels.

    Parameters
    ----------
    openface_dir : root directory of the parsed dataset with OpenFace
    emotion_dir : root directory of the CK+/AFEW dataset
    dataset : what is the dataset {CK, AFEW}
    data_type : which features to load {AUs, AU_activations, 2Dlandmarks}
    
    Returns
    -------
    Dict, Dict
        {Record identifier : {timestamp : {AU code : AU activation value}}},
        {Record identifier : Emotion identifier}
    """
    all_action_units = load_OpenFace_features(openface_dir, features=data_type)
    
    if dataset == 'CK':
        all_emotions = load_CK_emotions(emotion_dir)
    elif dataset == 'AFEW':
        all_emotions = load_AFEW_emotions(emotion_dir, set=set)
    else:
        ValueError("dataset argument must be either 'CK' or 'AFEW'")

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
    openface_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/afew_train_parsed")
    afew_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/afew")
    for data_type in ['AU_activations']:
        features, labels = load_data(openface_dir, afew_dir, dataset='AFEW', data_type=data_type, set='Train')
        labels = to_categorical(labels)

        openface_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/afew_val_parsed")
        test_features, test_labels = load_data(openface_dir, afew_dir, dataset='AFEW', data_type=data_type, set='Val')
        test_labels = to_categorical(test_labels)
        x_train, x_test = np.array(features), np.array(test_features)
        y_train, y_test = np.array(labels), np.array(test_labels)
        print(x_train[0])
        print(y_train[0])
        
        # Split into train and test sets
        #train_size = int(len(features) * 0.67)
        #test_size = len(features) - train_size
        #x_train, x_test = np.array(features[0:train_size]), np.array(features[train_size:len(features)])
        #y_train, y_test = np.array(labels[0:train_size]), np.array(labels[train_size:len(labels)])
        
        # Normalize length with zero-padding
        #maxlen = 71 # Maximum frames of a record from the Cohn-Kanade dataset
        #x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float32')
        #x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float32')
        for layers in [[100], [50,50], [15,30], [30,15], [100,100], [50,80], [80,50], [20,20,20], [100,100,100], [50,50,50], [20,50,100], [100,50,20]]:
            for epochs in [3, 6, 10, 20]:
                for batch_size in [1]:
                    print("Iteration parameters:", data_type, layers, epochs, batch_size)

                    # Create and fit the LSTM network
                    model = build_network(layers=layers, input_shape=(None,len(x_train[0][0])))
                    #model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)
                    for _ in range(epochs):
                        for seq, label in zip(x_train, y_train):
                            model.train_on_batch(np.array([seq]), np.array([label]))

                        # Final evaluation of the model
                        acc = 0
                        samples = 0
                        for seq, label in zip(x_test, y_test):
                            acc += model.test_on_batch(np.array([seq]), np.array([label]))[1]
                            samples += 1
                        print("Accuracy: %.2f%%" % (acc/samples*100))

                    #scores = model.evaluate(x_test, y_test, verbose=0)
                    #print("Accuracy: %.2f%%" % (scores[1]*100))
                    
if __name__ == "__main__":
    main()