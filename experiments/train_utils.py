# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - Training Utils
===============================================================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

import os
import dlib
import scipy.misc as sm
import matplotlib.pyplot as plt
import numpy as np
from  deepmotion.frontalization import facefrontal
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from keras.utils.np_utils import to_categorical

class FaceFrontalizer():
    """
    This is an adaptation from https://github.com/ChrisYang/facefrontalisation.
    I appologize if the code is not very clean, but there are certain
    implementation aspects that were not explained neither in his python
    conversion nor in the original matlab code.
    """
    def __init__(self):
        face_model_path = os.path.join(os.path.dirname(__file__), '../data/frontalization/face_shape.dat')
        ref3d_path = os.path.join(os.path.dirname(__file__), '../data/frontalization/ref3d.pkl')

        self.face_model = dlib.shape_predictor(face_model_path)
        self.face_detector = dlib.get_frontal_face_detector()
        self.frontalizer = facefrontal.frontalizer(ref3d_path)

    def frontalize_image(self, img):
        """
        Given a path to an image, it returns the same image frontalized.

        Parameters
        ----------
        filename : image file

        Returns
        -------
        PIL.Image
            Frontalized image
        """

        # Ask the detector to find the bounding boxes of each face. The ':,:,:3'
        # in the first argument avoids the last dimension in RGBA formats,
        # which dlib doesn't accept. The 1 in the second argument indicates that
        # we should upsample the image 1 time. This will make everything bigger
        # and allow us to detect more faces.
        detected_faces = self.face_detector(img[:, :, :3], 1)

        frontalized_faces = []
        for face in detected_faces:
            shape = self.face_model(img, face)
            p2d = np.asarray([(shape.part(n).x, shape.part(n).y,) for n in range(shape.num_parts)],
                             np.float32)

            rawfront, symfront = self.frontalizer.frontalization(img,face,p2d)
            frontalized_faces.append(sm.toimage(np.round(symfront).astype(np.uint8)))

        return frontalized_faces

    def frontalize_video(self, filename):
        cap = cv2.VideoCapture(filename)

        vid = []
        while True:
            ret, img = cap.read()
            if not ret:
                break
            frontalized_faces = frontalizer.frontalize(img)
            if len(frontalized_faces) > 0:
                # Just interested in one face per sample.
                img = np.array(frontalizer.frontalize(img)[0])
                vid.append(cv2.resize(img, (171, 128)))
        return np.array(vid, dtype=np.float32)

def class_labels(filename):
    """
    Returns the list of class labels from a file (one class per line)

    Parameters
    ----------
    filename : path to the file containing the class labels

    Returns
    -------
    List
        Class labels
    """
    with open(filename, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    return labels

def normalize(X, axis=0):
    """
    Given a list of samples, centers to the mean and component wise scales to
    unit variance.

    Parameters
    ----------
    X : list of samples
    axis : dimension to normalize

    Returns
    -------
    List
        Normalized list of samples
    """
    # Flatten the array and apply scaling
    flat_X = np.concatenate(X, axis=axis)
    flat_X = preprocessing.scale(flat_X)

    # Convert to original shape
    i, j = 0, 0
    while j < len(flat_X):
        X[i] = flat_X[j:j+len(X[i])]

        j += len(X[i])
        i += 1

    return X

def get_scores(model, features, labels):
    """ 
    Tests the prediction performance of a given model.

    Parameters
    ----------
    model : predictive model to test
    features : input data to the model
    labels : ground truth

    Returns
    -------
    List
        [model loss, model accuracy]
    """
    scores = []
    for X, Y in zip(features, labels):
        scores.append(model.test_on_batch(np.array([X]), np.array([Y])))

    return scores

def predict(model, features):
    """
    Given a model and a set of input features, returns the predictions.

    Parameters
    ----------
    model : predictive model to evaluate
    features : input data to the model

    Returns
    -------
    List
        Predictions
    """
    predictions = []
    for X in features:
        predictions.append(np.argmax(model.predict_on_batch(np.array([X]))))

    return predictions

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

class ModelEvaluator():
    """
    This class represents a wrapper used for evaluating a model through multiple iterations
    in Bayesian Optimization.
    """

    def __init__(self, get_model_method, x_train, x_test, y_train, y_test):
        self.get_model = get_model_method
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate(self, neurons, lr, lr_decay, epochs, batch_size=1):
        # The Gaussian Process' space is continous, so we need to round these values
        neurons, epochs, batch_size = map(lambda x: int(round(x)), (neurons, epochs, batch_size))

        # Create and fit the model
        model = self.get_model(layers=[neurons], lr=lr, lr_decay=lr_decay, input_shape=(None, len(self.x_train[0][0])))
        for _ in range(epochs):
            for X, Y in zip(self.x_train, self.y_train):
                model.train_on_batch(np.array([X]), np.array([Y]))

        # Evaluation using test data
        evals = get_scores(model, self.x_test, self.y_test)
        losses = [x[0] for x in evals]
        accuracies = [x[1] for x in evals]

        return np.mean(accuracies)

class KFoldEvaluator():
    """
    This class represents a wrapper used for evaluating a model through multiple iterations
    in Bayesian Optimization.
    """
    def __init__(self, get_model_method, features=None, labels=None):
        self.get_model = get_model_method
        self.features = features
        self.labels = labels

        # K-fold stratified cross-validation
        self.skf = StratifiedKFold(n_splits=10, shuffle=True)

    def evaluate(self, neurons, lr, lr_decay, epochs, batch_size=1):
        accuracies = []
        for train_index, test_index in self.skf.split(self.features, self.labels):
            x_train, x_test = [self.features[i] for i in train_index], [self.features[i] for i in test_index]
            y_train, y_test = to_categorical([self.labels[i] for i in train_index]), to_categorical([self.labels[i] for i in test_index])
            
            evaluator = ModelEvaluator(self.get_model, x_train, x_test, y_train, y_test)
            accuracies.append(evaluator.evaluate(neurons, lr, lr_decay, epochs, batch_size))
        
        return np.mean(accuracies)
