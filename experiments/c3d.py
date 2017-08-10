#!/usr/bin/env python

# C3D imports
import matplotlib
matplotlib.use('Agg')
from keras.models import model_from_json, Model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import deepmotion.c3d.c3d_model

import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.optimizers import Adam
import math
import datetime

# DeepMotion imports
from keras.layers import Dense, LSTM, Activation, BatchNormalization, TimeDistributed, Dropout, Input
from deepmotion.dataloader.ck_dataloader import load_CK_emotions, load_CK_videos
import train_utils

backend = K.image_dim_ordering
print("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
        backend))

input_size = (112, 112)
clip_length = 16

# mean to be subtracted
mean_dir = os.path.join(os.path.dirname(__file__), '../deepmotion/c3d/models/train01_16_128_171_mean.npy')
mean_cube = np.load(mean_dir)
mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))

frontalizer =  train_utils.face_frontalizer()

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
    List, List, List, List
        train video-paths, test video-paths, 
        CK+ train emotion labels, CK+ test emotion labels
    """
    videos = load_CK_videos(videos_path)
    emotions = load_CK_emotions(emotions_path)

    # Convert to training structures
    X, Y = dicts2lists(videos, emotions)
    Y = to_categorical(Y)

    # Split into train and test sets
    train_size = int(len(X) * 0.67)
    x_train, x_test = np.array(X[0:train_size]), np.array(X[train_size:len(X)])
    y_train, y_test = np.array(Y[0:train_size]), np.array(Y[train_size:len(Y)])

    return x_train, x_test, y_train, y_test

def get_features_model(c3d_model):
    
    # Remove last 4 layers, which are for sport classification
    for _ in range(4):
        c3d_model.layers.pop()
        c3d_model.outputs = [c3d_model.layers[-1].output]
        c3d_model.layers[-1].outbound_nodes = []

    c3d_model.compile(optimizer='sgd', loss='mse')

    return c3d_model

def get_temporal_model(summary=False, layers=[512], lr=0.001, lr_decay=0.0, input_shape=(1, None, 4096, )):

    input_features = Input(batch_shape=input_shape, name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(rate=0.5)(input_normalized)
    lstm = LSTM(layers[-1], return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    output_dropout = Dropout(rate=0.5)(lstm)
    #output = TimeDistributed(Dense(8, activation='softmax'), name='fc')(output_dropout)
    output = Dense(8, activation='softmax', name='fc')(output_dropout)

    temp_model = Model(inputs=input_features, outputs=output)

    adam_opt = Adam(lr=lr, decay=lr_decay)
    temp_model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

    if summary:
        print(model.summary())

    return temp_model

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
    model_dir = os.path.join(os.path.dirname(__file__), '../deepmotion/c3d/models')
    global backend

    epochs = 6
    batch_size = 1

    print("[Info] Using backend={}".format(backend))
    model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
    model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

    print("[Info] Reading model architecture...")
    model = model_from_json(open(model_json_filename, 'r').read())

    print("[Info] Loading model weights...")
    model.load_weights(model_weight_filename)
    print("[Info] Loading model weights -- DONE!")

    print("[Info] Loading labels...")
    labels_path = os.path.join(os.path.dirname(__file__), "../data/classification/labels.txt")
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))

    # DeepMotion
    features_model = get_features_model(model)
    temporal_model = get_temporal_model()

    # Load videos and emotions
    videos_path = os.path.join(os.path.dirname(__file__),  "../datasets/ck+videos")
    emotions_path = os.path.join(os.path.dirname(__file__), "../datasets/ck+")
    
    x_train, x_test, y_train, y_test = load_ck_data(videos_path, emotions_path)
    for _ in range(epochs):
        for X, Y in zip(x_train, y_train):
            X = parse_vid(X)
            Y = np.tile(Y,(len(X),1)).reshape((1,len(X),len(Y)))

            X = features_model.predict(X, batch_size=1, verbose=1)
            X = X.reshape((1,len(X),4096))
            
            temporal_model.train_on_batch(np.array(X), np.array(Y))
        
        # Final evaluation of the model
        acc = 0
        samples = 0
        for X, Y in zip(x_test, y_test):
            X = parse_vid(X)
            Y = np.tile(Y,(len(X),1)).reshape((1,len(X),len(Y)))

            X = features_model.predict(X, batch_size=1, verbose=1)
            X = X.reshape((1,len(X),4096))
            acc += temporal_model.test_on_batch(np.array(X), np.array(Y))[1]
            samples += 1
        print("Accuracy: %.2f%%" % (acc/samples*100))

if __name__ == '__main__':
    main()
