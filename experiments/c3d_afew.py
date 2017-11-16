# -*- coding: utf-8 -*-
"""
===============================================================================
Syna - Experimenting with C3D features
===============================================================================
"""
__author__ = "Daniyal Shahrokhian <daniyal@kth.se>"

import os

from bayes_opt import BayesianOptimization
from keras.utils.np_utils import to_categorical

import c3d_utils
import io_utils
import train_utils
from syna.dataloader.afew_dataloader import load_AFEW_data
from syna.syna_model import get_temporal_model


def main():
    # Load videos and emotions
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets/afew")
    x_train, y_train = load_AFEW_data(dataset_path, set='Train')
    x_test, y_test = load_AFEW_data(dataset_path, set='Val')

    # Extract features
    x_train, y_train = c3d_utils.clean_and_extract_C3D(x_train, y_train, remove_failures=True)
    x_test, y_test = c3d_utils.clean_and_extract_C3D(x_test, y_test)

    # Preprocess labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Bayesian Global Optimization of hyperparameters
    evaluator = train_utils.ModelEvaluator(get_temporal_model, x_train, x_test,
                                           y_train, y_test)
    hyper_opt = BayesianOptimization(evaluator.evaluate, {'neurons': (40, 200),
                                                          'epochs': (5, 100),
                                                          'lr': (0.0005, 0.005),
                                                          'lr_decay': (0.0, 1e-4),
                                                          'batch_size': (1, 1)})
    hyper_opt.maximize()
    optimal = hyper_opt.res['max']

    io_utils.report_metrics(get_temporal_model, optimal['max_params'], x_train,
                            x_test, y_train, y_test)

if __name__ == '__main__':
    main()
