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
from deepmotion.dataloader.ck_dataloader import load_CK_emotions
from deepmotion.dataloader.openface_dataloader import load_OpenFace_features
from deepmotion.deepmotion_model import get_temporal_model

import io_utils
import train_utils


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

def main():
    # Fix random seed for reproducibility
    np.random.seed(7)

    # Load the datasets
    features_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+norm")
    labels_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+")

    for feature_type in ['AU_activations', 'AUs', '2Dlandmarks']:
        print("Using " + feature_type)

        features, labels = load_ck_data(features_dir, labels_dir, feature_type=feature_type)
        features = train_utils.normalize(features)

        # Normalize length with zero-padding
        #maxlen = 71 # Maximum frames of a record from the Cohn-Kanade dataset
        #x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float32')
        #x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float32')

        # Bayesian Hyperparameter Optimization
        evaluator = train_utils.KFoldEvaluator(get_temporal_model, features, labels)
        hyper_opt = BayesianOptimization(evaluator.evaluate, {'neurons': (40, 200),
                                                              'epochs': (5, 100),
                                                              'lr': (0.0005, 0.005),
                                                              'lr_decay': (0.0, 1e-4),
                                                              'batch_size': (1, 1)
                                                             })
        hyper_opt.maximize()
        optimal = hyper_opt.res['max']

        print("Best hyperparameter settings: " + str(optimal))
        io_utils.kfold_report_metrics(get_temporal_model, optimal['max_params'], features, labels)

if __name__ == "__main__":
    main()
