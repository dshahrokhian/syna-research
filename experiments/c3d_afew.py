# -*- coding: utf-8 -*-
"""
===============================================================================
Syna - Experimenting with C3D features
===============================================================================
"""
__author__ = "Daniyal Shahrokhian <daniyal@kth.se>"

import math
import os

import cv2
import keras.backend as K
import matplotlib
import numpy as np
from bayes_opt import BayesianOptimization
from keras.layers import (LSTM, BatchNormalization, Dense, Dropout,
                          Input, TimeDistributed)
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import deepmotion.c3d.c3d_model
import io_utils
import train_utils
from deepmotion.dataloader.afew_dataloader import load_AFEW_data
from deepmotion.deepmotion_model import get_temporal_model

matplotlib.use('Agg')

backend = K.image_dim_ordering
print("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(backend))

input_size = (112, 112)
clip_length = 16

# mean to be subtracted
mean_dir = os.path.join(os.path.dirname(__file__),
                        '../deepmotion/c3d/models/train01_16_128_171_mean.npy')
mean_cube = np.load(mean_dir)
mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))

frontalizer = train_utils.FaceFrontalizer()

def get_feature_extractor(summary=False):
    model_dir = os.path.join(os.path.dirname(__file__), '../deepmotion/c3d/models')

    model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
    model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

    print("[Info] Reading model architecture...")
    model = model_from_json(open(model_json_filename, 'r').read())

    print("[Info] Loading model weights...")
    model.load_weights(model_weight_filename)
    print("[Info] Loading model weights -- DONE!")

    # Remove last 4 layers, which are for sport classification
    for _ in range(4):
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

    model.compile(optimizer='sgd', loss='mse')

    return model

def get_temporal_model2(summary=False, layers=[512], lr=0.001, lr_decay=0.0, input_shape=(1, None, 4096, )):

    input_features = Input(batch_shape=input_shape, name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(rate=0.5)(input_normalized)
    lstm = LSTM(layers[-1], return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    output_dropout = Dropout(rate=0.5)(lstm)
    output = TimeDistributed(Dense(8, activation='softmax'), name='fc')(output_dropout)

    model = Model(inputs=input_features, outputs=output)

    adam_opt = Adam(lr=lr, decay=lr_decay)
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

    if summary:
        print(model.summary())

    return model

def parse_vid(filename):
    cap = cv2.VideoCapture(filename)

    vid = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frontalized_faces = frontalizer.frontalize(img)
        if len(frontalized_faces) > 0:
            img = np.array(frontalizer.frontalize(img)[0]) # For now, just interested in one face per sample.
            vid.append(cv2.resize(img, (171, 128)))
    vid = np.array(vid, dtype=np.float32)

    if len(vid) == 0:
        return None

    n_frames = len(vid)
    n_clips = math.ceil(n_frames / clip_length)

    pad = np.zeros_like(vid[0], dtype=np.float32)

    n_missing_frames = (clip_length - (n_frames % clip_length)) % clip_length
    if n_missing_frames > 0:
        # Insert mean images (They will become blank images after subtraction)
        patch = np.array([pad,]*n_missing_frames)
        vid = np.concatenate((vid, patch), axis=0)

    # Subtract mean
    for i in range(0, n_clips*clip_length, clip_length):
        vid[i:i + clip_length] -= mean_cube
    # Should not be applied to inserted blank images
    if n_missing_frames > 0:
        vid[-n_missing_frames:] = np.sum([vid[-n_missing_frames:], mean_cube[-n_missing_frames:]], axis=0)

    # center crop;
    vid = vid[:, 8:120, 30:142, :] # (l, h, w, c)

    # Reshape into clips of length 'clip_length'
    vid = vid.reshape((n_clips, clip_length, 3, 112, 112))

    vid = vid.transpose(0, 1, 3, 4, 2)

    return vid

def clean_and_extract(features, labels):
    """
    Removes videos in which there are no detected faces, and then processes the rest by extracting
    C3D features.
    """
    feature_extractor = get_feature_extractor()
    videos = []

    for i in range(len(features)):
        parsed = parse_vid(features[i])
        if parsed is not None:
            videos.append(parsed)
        else:
            del labels[i]

    videos = [feature_extractor.predict(video, batch_size=1) for video in videos]
    videos = np.array(videos).reshape((1, len(videos), 4096))

    return videos, labels

def main():
    # Fix random seed for reproducibility
    np.random.seed(7)

    # Load videos and emotions
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets/afew")
    x_train, y_train = load_AFEW_data(dataset_path, set='Train')
    x_test, y_test = load_AFEW_data(dataset_path, set='Val')

    # Extract features
    x_train, y_train = clean_and_extract(x_train, y_train)
    x_test, y_test = clean_and_extract(x_test, y_test)
    
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
