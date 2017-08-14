#!/usr/bin/env python

import math
import os

import cv2
import keras.backend as K
import matplotlib
import numpy as np
from bayes_opt import BayesianOptimization
from keras.layers import (LSTM, Activation, BatchNormalization, Dense, Dropout,
                          Input, TimeDistributed)
from keras.models import Model, model_from_json
from keras.optimizers import Adam

import deepmotion.c3d.c3d_model
import io_utils
import train_utils
from deepmotion.dataloader.ck_dataloader import (load_CK_emotions,
                                                 load_CK_videos)
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

def dicts2lists(dict_videos, dict_emotions):
    """
    Converts the dictionaries of the dataloaders into lists, containing only
    records with identifiers present in both dictionaries, and ordered by
    record identifiers.

    Parameters
    ----------
    dict_videos : {Record identifier : video path}
    dict_emotions : {Record identifier : Emotion identifier}

    Returns
    -------
    List, List
        [videp paths], [emotion identifiers]
    """
    l_videos = []
    l_emotions = []

    for record_id, video_path in dict_videos.items():
        if record_id in dict_emotions:
            l_videos.append(video_path)
            l_emotions.append(dict_emotions[record_id])

    return l_videos, l_emotions

def load_ck_data(videos_path, emotions_path):
    """
    Extracts video paths and CK+ Emotion labels, preserving the order
    (e.g. x_train[0] corresponds to the same sample as y_train[0]).

    Parameters
    ----------
    videos_path : root directory of the parsed CK+ dataset containing videos
    emotion_path : root directory of the CK+ dataset

    Returns
    -------
    List, List
        video-paths, CK+ emotion labels
    """
    videos = load_CK_videos(videos_path)
    emotions = load_CK_emotions(emotions_path)

    return dicts2lists(videos, emotions)

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

        img = np.array(frontalizer.frontalize(img)[0]) # For now, just interested in one face per sample.

        vid.append(cv2.resize(img, (171, 128)))
    vid = np.array(vid, dtype=np.float32)

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

def main():
    # Fix random seed for reproducibility
    np.random.seed(7)

    # Load videos and emotions
    videos_path = os.path.join(os.path.dirname(__file__), "../datasets/ck+videos")
    emotions_path = os.path.join(os.path.dirname(__file__), "../datasets/ck+")
    videos, labels = load_ck_data(videos_path, emotions_path)
    
    # Extract features
    feature_extractor = get_feature_extractor()
    features = [parse_vid(video) for video in videos]
    features = [feature_extractor.predict(feat, batch_size=1) for feat in features] 
    features = features.reshape((1, len(features), 4096))

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
    # for X, Y in zip(x_train, y_train):
    #     X = parse_vid(X)
    #     Y = np.tile(Y, (len(X), 1)).reshape((1, len(X), len(Y)))

    #     X = features_model.predict(X, batch_size=1, verbose=1)
    #     X = X.reshape((1, len(X), 4096))

    #     temporal_model.train_on_batch(np.array(X), np.array(Y))

    # # Final evaluation of the model
    # acc = 0
    # samples = 0
    # for X, Y in zip(x_test, y_test):
    #     X = parse_vid(X)
    #     Y = np.tile(Y, (len(X), 1)).reshape((1, len(X), len(Y)))

    #     X = features_model.predict(X, batch_size=1, verbose=1)
    #     X = X.reshape((1, len(X), 4096))
    #     acc += temporal_model.test_on_batch(np.array(X), np.array(Y))[1]
    #     samples += 1
    # print("Accuracy: %.2f%%" % (acc/samples*100))

if __name__ == '__main__':
    main()
