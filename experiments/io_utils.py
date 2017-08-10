# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - Input/Output utils
===============================================================================
"""
__author__ = "Daniyal Shahrokhian <daniyal@kth.se>"

import csv
import datetime
import itertools
import math
import os

import deepmotion.deepmotion_model as deepmotion
import matplotlib.pyplot as plt
import numpy as np
import pandas
from bayes_opt import BayesianOptimization
from deepmotion.dataloader.ck_dataloader import load_CK_emotions
from deepmotion.dataloader.openface_dataloader import load_OpenFace_features
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from sklearn import metrics, preprocessing
from sklearn.model_selection import StratifiedKFold

import io_utils
import train_utils

def create_csv(filename, fields):
    with open(filename,'wb') as f:
        f.write(fields, delimiter=',')

def append_csv(filename, results):
    with open(filename, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(results)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def report_metrics(output_filename, get_model, hyperparams, features, labels):
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
        model = get_model(layers=[neurons], lr=hyperparams['lr'],
                                     lr_decay=hyperparams['lr_decay'],
                                     input_shape=(None,len(x_train[0][0])))
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
        y_pred.extend(train_utils.predict(model, x_test))
        y_true.extend(labels[test_index])

    losses = [x[0] for x in scores]
    accuracies = [x[1] for x in scores]
    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)

    # Print stats
    print("Test loss and Confidence Interval: %.2f (+/- %.2f)" % (np.mean(losses), np.std(losses)))
    print("Test accuracy and Confidence Interval: %.2f%% (+/- %.2f%%)"
          % (np.mean(accuracies)*100, np.std(accuracies)*100))
    print(cnf_matrix)
    print(metrics.classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Plot Confusion Matrix
    np.set_printoptions(precision=2)
    plt.figure()
    io_utils.plot_confusion_matrix(cnf_matrix, classes=CLASS_NAMES)
    plt.show()
