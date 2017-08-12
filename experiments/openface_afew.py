# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - Example with OpenFace network
===============================================================================
"""
__author__ = "Daniyal Shahrokhian <daniyal@kth.se>"

import os

import deepmotion.deepmotion_model as deepmotion
import numpy as np
from bayes_opt import BayesianOptimization
from deepmotion.dataloader.afew_dataloader import load_AFEW_emotions
from deepmotion.dataloader.openface_dataloader import load_OpenFace_features
from keras.utils.np_utils import to_categorical

import io_utils
import train_utils


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
    train_action_units = load_OpenFace_features(os.path.join(openface_dir, 'Train'), 
                                                features=feature_type)
    train_emotions = load_AFEW_emotions(emotion_dir, set='Train')
    train_features, train_labels = train_utils.dicts2lists(train_action_units,
                                                           train_emotions)
    train_labels = to_categorical(train_labels)

    test_action_units = load_OpenFace_features(os.path.join(openface_dir, 'Val'),
                                               features=feature_type)
    test_emotions = load_AFEW_emotions(emotion_dir, set='Val')
    test_features, test_labels = train_utils.dicts2lists(test_action_units, test_emotions)
    test_labels = to_categorical(test_labels)

    x_train, x_test = np.array(train_features), np.array(test_features)
    y_train, y_test = np.array(train_labels), np.array(test_labels)

    return x_train, x_test, y_train, y_test

def main():
    # Fix random seed for reproducibility
    np.random.seed(7)

    # Load the datasets
    features_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/afew_parsed")
    labels_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/afew")

    for feature_type in ['AU_activations', 'AUs', '2Dlandmarks']:
        x_train, x_test, y_train, y_test = load_afew_data(features_dir, labels_dir,
                                                          feature_type=feature_type)
        x_train = train_utils.normalize(x_train)
        x_test = train_utils.normalize(x_test)

        # Normalize length with zero-padding
        #maxlen = 71 # Maximum frames of a record from the Cohn-Kanade dataset
        #x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float32')
        #x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float32')

        # Bayesian Global Optimization of hyperparameters
        evaluator = train_utils.ModelEvaluator(deepmotion.get_temporal_model, x_train, x_test,
                                               y_train, y_test)
        hyper_opt = BayesianOptimization(evaluator.evaluate, {'neurons': (40, 200),
                                                              'epochs': (1, 20),
                                                              'lr': (0.0001, 0.01),
                                                              'lr_decay': (0.0, 1e-4),
                                                              'batch_size': (1, 1)})
        hyper_opt.maximize()
        optimal = hyper_opt.res['max']

        print("Best hyperparameter settings: " + str(optimal))
        io_utils.report_metrics(deepmotion.get_temporal_model, optimal['max_params'], x_train,
                                x_test, y_train, y_test)

if __name__ == "__main__":
    main()
