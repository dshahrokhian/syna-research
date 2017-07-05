# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - Example with OpenFace network
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
import deepmotion.deepmotion_model as deepmotion
from bayes_opt import BayesianOptimization

# Data imports
from deepmotion.dataloader.openface_dataloader import load_OpenFace_features
from deepmotion.dataloader.ck_dataloader import load_CK_emotions
import train_utils
import io_utils

openface_feature = ''
features, labels = None, None

class_labels_filename = os.path.join(os.path.dirname(__file__), "../data/classification/labels.txt")

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

def report_metrics(output_filename, hyperparams):
    io_utils.append_csv(output_filename, ['feature_type',
                                        'epoch', 
                                        'train_loss', 
                                        'test_loss', 
                                        'train_acc', 
                                        'test_acc'])

    # The Gaussian Process' space is continous, so we need to round some values
    neurons, epochs, batch_size = map(lambda x: int(round(x)), 
        (hyperparams['neurons'], hyperparams['epochs'], hyperparams['batch_size']))

    # K-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True)

    scores, y_true, y_pred = [], [], []
    for train_index, test_index in skf.split(features, labels):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = to_categorical(labels[train_index]), to_categorical(labels[test_index]) 

        # For visualization purposes, we will report training metrics 100 times.
        report_freq = int(epochs*len(x_train)/100)

        # Create and fit the LSTM network
        model = deepmotion.get_model(layers=[neurons], lr=hyperparams['lr'], lr_decay=hyperparams['lr_decay'], input_shape=(None,len(x_train[0][0])))
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
        accuracies = [x[1] for x in evals]
        scores.append([np.mean(losses), np.mean(accuracies)])

        # Store predictions and ground truths
        y_pred.extend(train_utils.predict(x_test))
        y_true.extend(y_test)

    losses = [x[0] for x in scores]
    accuracies = [x[1] for x in scores]

    print("Test loss and Confidence Interval: %.2f (+/- %.2f)" % (np.mean(losses), np.std(losses)))
    print("Test accuracy and Confidence Interval: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies)*100, np.std(accuracies)*100))
    print(metrics.confusion_matrix(y_true, y_pred))
    print(metrics.classification_report(true_classes, predicted_classes, target_names=train_utils.class_labels(class_labels_filename)))

def adam_evaluate(neurons, lr, lr_decay, epochs, batch_size):
    # The Gaussian Process' space is continous, so we need to round some values
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
        evals = train_utils.evaluate(model, x_test, y_test)
        losses = [x[0] for x in evals]
        accuracies = [x[1] for x in evals]
        scores.append([np.mean(losses), np.mean(accuracies)])

    losses = [x[0] for x in scores]
    accuracies = [x[1] for x in scores]

    print("Test loss and Standard dev: %.2f (+/- %.2f)" % (np.mean(losses), np.std(losses)))
    print("Test accuracy and Standard dev: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies)*100, np.std(accuracies)*100))

    return np.mean(accuracies)

def main():
    # Load the datasets
    features_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+norm")
    labels_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+")
    
    for feature_type in ['AU_activations', 'AUs', '2Dlandmarks']:
        global openface_feature
        openface_feature = feature_type
        
        # Fix random seed for reproducibility
        np.random.seed(7)
        
        global features, labels
        features, labels = load_ck_data(features_dir, labels_dir, feature_type=feature_type)
        features = train_utils.normalize(features)

        # Normalize length with zero-padding
        #maxlen = 71 # Maximum frames of a record from the Cohn-Kanade dataset
        #x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float32')
        #x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float32')

        # Bayesian Hyperparameter Optimization
        hyper_opt = BayesianOptimization(adam_evaluate, {'neurons': (40, 200),
                                                    'epochs': (5, 40),
                                                    'lr': (0.0005, 0.005),
                                                    'lr_decay': (0.0, 1e-4),
                                                    'batch_size': (1, 1)
                                                    })
        hyper_opt.maximize()
        optimal = hyper_opt.res['max']
        
        print("Best hyperparameter settings: " + str(optimal))
        report_metrics('results.csv', optimal['max_params'])
        
if __name__ == "__main__":
    main()