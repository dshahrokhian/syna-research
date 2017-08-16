# -*- coding: utf-8 -*-
"""
===============================================================================
Syna - Experimenting with OpenFace features
===============================================================================
"""
__author__ = "Daniyal Shahrokhian <daniyal@kth.se>"

import os

import numpy as np
from bayes_opt import BayesianOptimization
from keras.utils.np_utils import to_categorical

import io_utils
import train_utils
from deepmotion.dataloader.afew_dataloader import load_AFEW_emotions
from deepmotion.dataloader.openface_dataloader import load_OpenFace_features
from deepmotion.deepmotion_model import get_temporal_model


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
    train_features = load_OpenFace_features(os.path.join(openface_dir, 'Train'),
                                            features=feature_type)
    train_labels = load_AFEW_emotions(emotion_dir, set='Train')
    train_features, train_labels = train_utils.dicts2lists(train_features,
                                                           train_labels)

    test_features = load_OpenFace_features(os.path.join(openface_dir, 'Val'),
                                           features=feature_type)
    test_labels = load_AFEW_emotions(emotion_dir, set='Val')
    test_features, test_labels = train_utils.dicts2lists(test_features, test_labels)

    return train_features, train_labels, test_features, test_labels

def main():
    # Fix random seed for reproducibility
    np.random.seed(7)

    # Load the datasets
    features_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/afew_parsed")
    labels_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/afew")

    for feature_type in ['AUs', '2Dlandmarks']:
        print("Using " + feature_type)
        
        x_train, y_train, x_test, y_test = load_afew_data(features_dir, labels_dir,
                                                          feature_type=feature_type)
        x_train = train_utils.normalize(x_train)
        x_test = train_utils.normalize(x_test)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # Normalize length with zero-padding
        #maxlen = 71 # Maximum frames of a record from the Cohn-Kanade dataset
        #x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float32')
        #x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float32')

        # Bayesian Global Optimization of hyperparameters
        evaluator = train_utils.ModelEvaluator(get_temporal_model, x_train, x_test,
                                               y_train, y_test)
        hyper_opt = BayesianOptimization(evaluator.evaluate, {'neurons': (40, 200),
                                                              'epochs': (5, 40),
                                                              'lr': (0.0005, 0.005),
                                                              'lr_decay': (0.0, 1e-4)})
        hyper_opt.maximize()
        optimal = hyper_opt.res['max']

        print("Best hyperparameter settings: " + str(optimal))
        io_utils.report_metrics(get_temporal_model, optimal['max_params'], x_train,
                                x_test, y_train, y_test)

if __name__ == "__main__":
    main()
