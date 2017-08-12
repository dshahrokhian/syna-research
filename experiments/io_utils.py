# -*- coding: utf-8 -*-
"""
===============================================================================
Syna - Input/Output utils
===============================================================================
"""
__author__ = "Daniyal Shahrokhian <daniyal@kth.se>"

import csv
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn import metrics
from sklearn.learning_curve import learning_curve
from sklearn.model_selection import StratifiedKFold

import train_utils

CLASS_NAMES = train_utils.class_labels(os.path.join(os.path.dirname(__file__),
                                                    "../data/classification/labels.txt"))

def create_csv(filename, fields):
    with open(filename, 'wb') as f:
        f.write(fields, delimiter=',')

def append_csv(filename, results):
    with open(filename, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(results)

def report_metrics(get_model, hyperparams, x_train, x_test, y_train, y_test):
    # The Gaussian Process' space is continous, so we need to round some values
    neurons, epochs, batch_size = map(lambda x: int(round(x)),
                                      (hyperparams['neurons'], hyperparams['epochs'],
                                       hyperparams['batch_size']))

    # Create and fit the model
    model = get_model(layers=[neurons], lr=hyperparams['lr'],
                      lr_decay=hyperparams['lr_decay'],
                      input_shape=(None, len(x_train[0][0])))

    history = model.fit(x_train, y_train, batch_size=1, epochs=epochs, callbacks=None, 
                        validation_data=(x_test, y_test))

    # Final evaluation of the model
    evals = train_utils.get_scores(model, x_test, y_test)
    losses = [x[0] for x in evals]
    accuracies = [x[1] for x in evals]
    scores = [np.mean(losses), np.mean(accuracies)]

    # Store predictions
    y_pred = train_utils.predict(model, x_test)
    y_true = [np.argmax(y) for y in y_test]

    losses = [x[0] for x in scores]
    accuracies = [x[1] for x in scores]
    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)

    # Print stats
    print("Test loss and Confidence Interval: %.2f (+/- %.2f)" % (np.mean(losses), np.std(losses)))
    print("Test accuracy and Confidence Interval: %.2f%% (+/- %.2f%%)"
          % (np.mean(accuracies)*100, np.std(accuracies)*100))
    print(metrics.classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Plot figures
    plot_confusion_matrix(cnf_matrix, classes=CLASS_NAMES)
    plot_learning_curve(get_model(layers=[neurons], lr=hyperparams['lr'],
                                  lr_decay=hyperparams['lr_decay'],
                                  input_shape=(None, len(x_train[0][0]))),
                        x_train, y_train, title="Learning Curve")
    plot_model_training(history['loss'], history['val_loss'], history['acc'], history['val_acc'])
    plt.show()

def kfold_report_metrics(get_model, hyperparams, features, labels):
    # The Gaussian Process' space is continous, so we need to round some values
    neurons, epochs, batch_size = map(lambda x: int(round(x)),
                                      (hyperparams['neurons'], hyperparams['epochs'],
                                       hyperparams['batch_size']))

    # K-fold stratified cross-validation
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    train_losses, train_accs = np.zeros(epochs), np.zeros(epochs)
    test_losses, test_accs = np.zeros(epochs), np.zeros(epochs)
    scores, y_true, y_pred = [], [], []
    for train_index, test_index in skf.split(features, labels):
        x_train, x_test = [features[i] for i in train_index], [features[i] for i in test_index]
        y_train = to_categorical([labels[i] for i in train_index])
        y_test = to_categorical([labels[i] for i in test_index])

        # Create and fit the model
        model = get_model(layers=[neurons], lr=hyperparams['lr'],
                          lr_decay=hyperparams['lr_decay'],
                          input_shape=(None, len(x_train[0][0])))

        for epoch in range(epochs):
            for X, Y in zip(x_train, y_train):
                model.train_on_batch(np.array([X]), np.array([Y]))

            train_evals = train_utils.get_scores(model, x_train, y_train)
            test_evals = train_utils.get_scores(model, x_test, y_test)

            train_losses[epoch] += np.mean([x[0] for x in train_evals])
            test_losses[epoch] += np.mean([x[0] for x in test_evals])
            train_accs[epoch] += np.mean([x[1] for x in train_evals])
            test_accs[epoch] += np.mean([x[1] for x in test_evals])

        # Final evaluation of the model
        evals = train_utils.get_scores(model, x_test, y_test)
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
    print(metrics.classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Plot figures
    plot_confusion_matrix(cnf_matrix, classes=CLASS_NAMES)
    plot_learning_curve(get_model(layers=[neurons], lr=hyperparams['lr'],
                                  lr_decay=hyperparams['lr_decay'],
                                  input_shape=(None, len(x_train[0][0]))),
                        features, labels, title="10-Fold Learning Curve", cv=n_splits)
    plot_model_training(train_losses/n_splits, test_losses/n_splits, train_accs/n_splits,
                        test_accs/n_splits)
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)
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
    np.set_printoptions(precision=8)

def plot_learning_curve(estimator, X, y, title="Learning Curves", ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_model_training(loss, val_loss, acc, val_acc):
    # Losses
    plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    # Accuracies
    plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.show()
