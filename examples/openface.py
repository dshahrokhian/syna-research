# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - Example with OpenFace network
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
from sklearn.metrics import mean_squared_error
from keras.utils.np_utils import to_categorical
from deepmotion.deepmotion_model import get_model

# Data Loader imports
from deepmotion.dataloader.openface_dataloader import load_OpenFace_features
from deepmotion.dataloader.ck_dataloader import load_CK_emotions
from deepmotion.dataloader.afew_dataloader import load_AFEW_emotions

def dicts2lists(dict_features, dict_emotions):
    """ 
    Converts the dictionaries of the dataloaders into lists, containing only
    records with identifiers present in both dictionaries, and ordered by 
    record identifiers.
    
    Parameters
    ----------
    dict_features : {Record identifier : 
                            {timestamp :    
                                {feature code : feature value}
                            }
                        }
    dict_emotions : {Record identifier : Emotion identifier}
    
    Returns
    -------
    List, List
        [records, samples, features], [emotion identifiers]
    """
    l_features = []
    l_emotions = []

    for record_id, values in dict_features.items():
        if record_id in dict_emotions:
            record_features = []
            for timestamp in sorted(values.keys()):
                record_features.append(list(values[timestamp].values()))
            
            l_features.append(record_features)
            l_emotions.append(dict_emotions[record_id])
    
    return l_features, l_emotions


def load_ck_data(openface_dir, emotion_dir, data_type='AUs', split=0.67):
    """ 
    Extracts OpenFace Action Units features and CK+ Emotion labels,
    preserving the order (e.g. x_train[0] corresponds to the same sample as
    y_train[0]).

    Parameters
    ----------
    openface_dir : root directory of the parsed dataset with OpenFace
    emotion_dir : root directory of the CK+ dataset
    data_type : which features to load {AUs, AU_activations, 2Dlandmarks}
    split : Training set split ratio [0,1] (1-split will be returned as Testing set)
    
    Returns
    -------
    List, List, List, List
        OpenFace train features, OpenFace test features, 
        CK+ train emotion labels, CK+ test emotion labels
    """
    features = load_OpenFace_features(openface_dir, features=data_type)
    labels = load_CK_emotions(emotion_dir)

    # Convert to training structures
    X, Y = dicts2lists(features, labels)
    Y = to_categorical(Y)

    # Split into train and test sets
    train_size = int(len(X) * split)
    x_train, x_test = np.array(X[0:train_size]), np.array(X[train_size:len(X)])
    y_train, y_test = np.array(Y[0:train_size]), np.array(Y[train_size:len(Y)])

    return x_train, x_test, y_train, y_test

def load_afew_data(openface_dir, emotion_dir, data_type='AUs'):
    """ 
    Extracts OpenFace Action Units features and AFEW Emotion labels,
    preserving the order (e.g. x_train[0] corresponds to the same sample as
    y_train[0]).

    Parameters
    ----------
    openface_dir : root directory of the parsed dataset with OpenFace
    emotion_dir : root directory of the AFEW dataset
    data_type : which features to load {AUs, AU_activations, 2Dlandmarks}
    
    Returns
    -------
    List, List, List, List
        OpenFace train features, OpenFace test features, 
        AFEW train emotion labels, AFEW test emotion labels
    """
    train_action_units = load_OpenFace_features( os.path.join(openface_dir, 'Train'), features=data_type )
    train_emotions = load_AFEW_emotions(emotion_dir, set='Train')
    train_features, train_labels = dicts2lists(train_action_units, train_emotions)
    train_labels = to_categorical(train_labels)

    test_action_units = load_OpenFace_features( os.path.join(openface_dir, 'Val'), features=data_type )
    test_emotions = load_AFEW_emotions(emotion_dir, set='Val')
    test_features, test_labels = dicts2lists(test_action_units, test_emotions)
    test_labels = to_categorical(test_labels)

    x_train, x_test = np.array(train_features), np.array(test_features)
    y_train, y_test = np.array(train_labels), np.array(test_labels)

    return x_train, x_test, y_train, y_test

def main():
    # Fix random seed for reproducibility
    np.random.seed(7)
    
    # Load the datasets
    features_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+parsed")
    labels_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+")
    
    for data_type in ['AU_activations']:
        x_train, x_test, y_train, y_test = load_ck_data(features_dir, labels_dir, data_type=data_type)
        
        # Normalize length with zero-padding
        #maxlen = 71 # Maximum frames of a record from the Cohn-Kanade dataset
        #x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float32')
        #x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float32')
        for layers in [[100], [50,50], [15,30], [30,15], [100,100], [50,80], [80,50], [20,20,20], [100,100,100], [50,50,50], [20,50,100], [100,50,20]]:
            for epochs in [3, 6, 10, 20]:
                for batch_size in [1]:
                    print("Iteration parameters:", data_type, layers, epochs, batch_size)

                    # Create and fit the LSTM network
                    model = get_model(layers=layers, input_shape=(None,len(x_train[0][0])))
                    #model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)
                    for _ in range(epochs):
                        for X, Y in zip(x_train, y_train):
                            model.train_on_batch(np.array([X]), np.array([Y]))

                        # Final evaluation of the model
                        acc = 0
                        samples = 0
                        for X, Y in zip(x_test, y_test):
                            acc += model.test_on_batch(np.array([X]), np.array([Y]))[1]
                            samples += 1
                        print("Accuracy: %.2f%%" % (acc/samples*100))

                    #scores = model.evaluate(x_test, y_test, verbose=0)
                    #print("Accuracy: %.2f%%" % (scores[1]*100))
                    
if __name__ == "__main__":
    main()