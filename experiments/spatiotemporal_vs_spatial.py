# -*- coding: utf-8 -*-
"""
===============================================================================
Syna - Spatial vs Spatiotemporal Emotion Recognition
===============================================================================
"""
__author__ = "Daniyal Shahrokhian <daniyal@kth.se>"

import os
import datetime

# Neural Network imports
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import metrics
from keras.utils.np_utils import to_categorical
import syna.syna_model as syna
from bayes_opt import BayesianOptimization

# Data imports
from syna.dataloader.openface_dataloader import load_OpenFace_features
from syna.dataloader.ck_dataloader import load_CK_emotions

import train_utils
import io_utils

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import Adam

openface_feature = ''
features, labels = None, None

CLASS_NAMES = train_utils.class_labels(os.path.join(os.path.dirname(__file__), "../data/classification/labels.txt"))

def get_spatial_model(summary=False, layers=[100], lr=0.001, lr_decay=0.0, input_shape=(10, 64)):
    """ 
    Returns a 2-fully-connected network.
    
    Parameters
    ----------
    summary : print model summary
    layers : list with the number of LSTM units per layer
    lr : learning rate
    lr_decay : learning rate decay
    input_shape : input shape to the network
    
    Returns
    -------
    Sequential
        Keras model
    """
    model = Sequential()

    model.add(Dense(layers[-1], activation='tanh', input_shape=input_shape))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    
    adam_opt = Adam(lr=lr, decay=lr_decay)
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

    if summary:
        print(model.summary())

    return model

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

    return train_utils.dicts2lists(features, labels)    

def adam_evaluate(neurons, lr, lr_decay, epochs, batch_size):
    # The Gaussian Process' space is continous, so we need to round some values
    neurons, epochs, batch_size = map(lambda x: int(round(x)), (neurons, epochs, batch_size))

    # K-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    
    scores = []
    for train_index, test_index in skf.split(features, labels):
        x_train, x_test = [features[i] for i in train_index], [features[i] for i in test_index]
        y_train, y_test = to_categorical([labels[i] for i in train_index]), to_categorical([labels[i] for i in test_index])

        # Create and fit the LSTM network
        model = get_spatial_model(layers=[neurons], lr=lr, lr_decay=lr_decay, input_shape=(len(x_train[0]),))
        for _ in range(epochs):
            for X, Y in zip(x_train, y_train):
                model.train_on_batch(np.array([X]), np.array([Y]))

        # Final evaluation of the model
        evals = train_utils.evaluate(model, x_test, y_test)
        losses = [x[0] for x in evals]
        accuracies = [x[1] for x in evals]
        scores.append([np.mean(losses), np.mean(accuracies)])

    losses = [x[0] for x in scores]
    accuracies = [x[1] for x in scores]

    print("Test loss and Standard dev: %.2f (+/- %.2f)" % (np.mean(losses), np.std(losses)))
    print("Test accuracy and Standard dev: %.2f%% (+/- %.2f%%)" 
          % (np.mean(accuracies)*100, np.std(accuracies)*100))

    return np.mean(accuracies)

def main():
    # Load the datasets
    features_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+norm")
    labels_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+")
    
    for feature_type in ['2Dlandmarks']:
        global openface_feature
        openface_feature = feature_type
        
        # Fix random seed for reproducibility
        np.random.seed(7)
        
        global features, labels
        features, labels = load_ck_data(features_dir, labels_dir, feature_type=feature_type)
        features = train_utils.normalize(features)
        features = [feat[-1] for feat in features] # Include only last image as emotion


        # Normalize length with zero-padding
        #maxlen = 71 # Maximum frames of a record from the Cohn-Kanade dataset
        #x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float32')
        #x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float32')

        # Bayesian Hyperparameter Optimization
        hyper_opt = BayesianOptimization(adam_evaluate, {'neurons': (8, 40),
                                                         'epochs': (5, 40),
                                                         'lr': (0.0005, 0.005),
                                                         'lr_decay': (0.0, 1e-4),
                                                         'batch_size': (1, 1)
                                                        })
        hyper_opt.maximize()
        optimal = hyper_opt.res['max']
        
        print("Best hyperparameter settings: " + str(optimal))
        io_utils.report_metrics('results_spatial.csv', syna.get_temporal_model, optimal['max_params'], features, labels)
        
if __name__ == "__main__":
    main()