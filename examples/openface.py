# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - Example with OpenFace network
===============================================================================
"""
__author__ = "Daniyal Shahrokhian <daniyal@kth.se>"

# Neural Network imports
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
import deepmotion.deepmotion_model as deepmotion
from bayes_opt import BayesianOptimization

# Data Loader imports
from deepmotion.dataloader.openface_dataloader import load_OpenFace_features
from deepmotion.dataloader.ck_dataloader import load_CK_emotions
from deepmotion.dataloader.afew_dataloader import load_AFEW_emotions
import train_utils

features, labels = None, None

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
    
    return np.array(l_features), np.array(l_emotions)


def load_ck_data(openface_dir, emotion_dir, feature_type='AUs'):
    """ 
    Extracts OpenFace Action Units features and CK+ Emotion labels,
    preserving the order (e.g. x_train[0] corresponds to the same sample as
    y_train[0]).

    Parameters
    ----------
    openface_dir : root directory of the parsed dataset with OpenFace
    emotion_dir : root directory of the CK+ dataset
    feature_type : which features to load {AUs, AU_activations, 2Dlandmarks}
    
    Returns
    -------
    List, List
        OpenFace features, CK+ emotion labels
    """
    features = load_OpenFace_features(openface_dir, features=feature_type)
    labels = load_CK_emotions(emotion_dir)

    return dicts2lists(features, labels)

def load_afew_data(openface_dir, emotion_dir, feature_type='AUs'):
    """ 
    Extracts OpenFace Action Units features and AFEW Emotion labels,
    preserving the order (e.g. x_train[0] corresponds to the same sample as
    y_train[0]).

    Parameters
    ----------
    openface_dir : root directory of the parsed dataset with OpenFace
    emotion_dir : root directory of the AFEW dataset
    feature_type : which features to load {AUs, AU_activations, 2Dlandmarks}
    
    Returns
    -------
    List, List, List, List
        OpenFace train features, OpenFace test features, 
        AFEW train emotion labels, AFEW test emotion labels
    """
    train_action_units = load_OpenFace_features( os.path.join(openface_dir, 'Train'), features=feature_type )
    train_emotions = load_AFEW_emotions(emotion_dir, set='Train')
    train_features, train_labels = dicts2lists(train_action_units, train_emotions)

    test_action_units = load_OpenFace_features( os.path.join(openface_dir, 'Val'), features=feature_type )
    test_emotions = load_AFEW_emotions(emotion_dir, set='Val')
    test_features, test_labels = dicts2lists(test_action_units, test_emotions)

    x_train, x_test = np.array(train_features), np.array(test_features)
    y_train, y_test = np.array(train_labels), np.array(test_labels)

    return x_train, x_test, y_train, y_test

def adam_evaluate(neurons, lr, lr_decay, epochs, batch_size):
    print("New evaluation")
    print(datetime.datetime.now())
    # The Gaussian Process' space is continous, so we need to round these values
    neurons, epochs, batch_size = map(lambda x: int(round(x)), (neurons, epochs, batch_size))

    # K-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True)

    scores = []
    for train_index, test_index in skf.split(features, labels):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = to_categorical(labels[train_index]), to_categorical(labels[test_index]) 

        # Create and fit the LSTM network
        model = deepmotion.get_model(layers=[neurons], lr=lr, lr_decay=lr_decay, input_shape=(None,len(x_train[0][0])))

        for _ in range(epochs):
            for X, Y in zip(x_train, y_train):
                model.train_on_batch(np.array([X]), np.array([Y]))

        # Final evaluation of the model
        accuracies = []
        for X, Y in zip(x_test, y_test):
            accuracies.append(model.test_on_batch(np.array([X]), np.array([Y]))[1])
        
        acc = np.mean(accuracies)
        scores.append(acc)
        
    print("Accuracy and Standard dev: %.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
    print(datetime.datetime.now())
    
    return np.mean(scores)

def main():
    # Load the datasets
    features_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+norm")
    labels_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+")
    
    for feature_type in ['AU_activations', 'AUs', '2Dlandmarks']:
        print("Using " + feature_type)
        
        # Fix random seed for reproducibility
        np.random.seed(7)
        
        global features, labels
        features, labels = load_ck_data(features_dir, labels_dir, feature_type=feature_type)
        features = train_utils.normalize(features) # Input normalization

        # Normalize length with zero-padding
        #maxlen = 71 # Maximum frames of a record from the Cohn-Kanade dataset
        #x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float32')
        #x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float32')

        # Bayesian Hyperparameter Optimization
        hyper_opt = BayesianOptimization(adam_evaluate, {'neurons': (20, 100),
                                                    'epochs': (1, 20),
                                                    'lr': (0.0001, 0.01),
                                                    'lr_decay': (0.0, 1e-4),
                                                    'batch_size': (1, 1)
                                                    })
        print("start")
        print(datetime.datetime.now())
        hyper_opt.maximize()
        print("finish")
        print(datetime.datetime.now())
        
        print("Best hyperparameter settings: " + str(hyper_opt.res['max']))        
                    
if __name__ == "__main__":
    main()