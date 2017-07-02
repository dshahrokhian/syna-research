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

# Data imports
from deepmotion.dataloader.openface_dataloader import load_OpenFace_features
from deepmotion.dataloader.afew_dataloader import load_AFEW_emotions
import train_utils
import io_utils

output_filename = 'results.csv'
openface_feature = ''

x_train, x_test, y_train, y_test = None, None, None, None

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
    train_features, train_labels = train_utils.dicts2lists(train_action_units, train_emotions)
    train_labels = to_categorical(train_labels)

    test_action_units = load_OpenFace_features( os.path.join(openface_dir, 'Val'), features=feature_type )
    test_emotions = load_AFEW_emotions(emotion_dir, set='Val')
    test_features, test_labels = train_utils.dicts2lists(test_action_units, test_emotions)
    test_labels = to_categorical(test_labels)

    x_train, x_test = np.array(train_features), np.array(test_features)
    y_train, y_test = np.array(train_labels), np.array(test_labels)

    return x_train, x_test, y_train, y_test

def adam_evaluate(neurons, lr, lr_decay, epochs, batch_size):
    # The Gaussian Process' space is continous, so we need to round these values
    neurons, epochs, batch_size = map(lambda x: int(round(x)), (neurons, epochs, batch_size))

    # For visualization purposes, we will report training results 100 times.
    report_freq = epochs*len(x_train)/100

    # Create and fit the LSTM network
    model = deepmotion.get_model(layers=[neurons], lr=lr, lr_decay=lr_decay, input_shape=(None,len(x_train[0][0])))
    i = 0
    for epoch in range(epochs):
        for X, Y in zip(x_train, y_train):
            model.train_on_batch(np.array([X]), np.array([Y]))

            if i % report_freq == 0:
                train_evals = train_utils.evaluate(model, x_train, y_train)
                test_evals = train_utils.evaluate(model, x_test, y_test)

                io_utils.append_csv(output_filename, [openface_feature,
                                                        epoch, 
                                                        np.mean([x[0] for x in train_evals]), 
                                                        np.mean([x[0] for x in test_evals]), 
                                                        np.mean([x[1] for x in train_evals])*100, 
                                                        np.mean([x[1] for x in test_evals])*100])
            i += 1
    
    # Final evaluation of the model
    evals = train_utils.evaluate(model, x_test, y_test)

    losses = [x[0] for x in evals]
    accuracies = [x[1]*100 for x in evals]

    print("Test accuracy and Standard dev: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies), np.std(accuracies)))
    print("Test loss and Standard dev: %.2f%% (+/- %.2f%%)" % (np.mean(losses), np.std(losses)))
    
    return np.mean(losses)

def main():
    # Load the datasets
    features_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/afew_parsed")
    labels_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/afew")
    
    for feature_type in ['AU_activations', 'AUs', '2Dlandmarks']:
        global openface_feature
        openface_feature = feature_type
        
        # Fix random seed for reproducibility
        np.random.seed(7)
        
        global x_train, x_test, y_train, y_test
        x_train, x_test, y_train, y_test = load_afew_data(features_dir, labels_dir, feature_type=feature_type)
        x_train = train_utils.normalize(x_train)
        x_test = train_utils.normalize(x_test)

        # Normalize length with zero-padding
        #maxlen = 71 # Maximum frames of a record from the Cohn-Kanade dataset
        #x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float32')
        #x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float32')

        # Bayesian Global Optimization of hyperparameters
        hyper_opt = BayesianOptimization(adam_evaluate, {'neurons': (40, 200),
                                                    'epochs': (1, 20),
                                                    'lr': (0.0001, 0.01),
                                                    'lr_decay': (0.0, 1e-4),
                                                    'batch_size': (1, 1)
                                                    })
        hyper_opt.maximize()
        
        print("Best hyperparameter settings: " + str(hyper_opt.res['max']))        
                    
if __name__ == "__main__":
    io_utils.append_csv(output_filename, ['feature_type',
                                        'epoch', 
                                        'train_loss', 
                                        'test_loss', 
                                        'train_acc', 
                                        'test_acc'])
    main()