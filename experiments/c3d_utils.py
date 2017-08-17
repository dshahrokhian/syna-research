# -*- coding: utf-8 -*-
"""
===============================================================================
Syna - C3D Pre-processing utils
===============================================================================
"""
__author__ = "Daniyal Shahrokhian <daniyal@kth.se>"

import matplotlib
matplotlib.use('Agg')

import math
import os

import cv2

import cv2
import keras.backend as K

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

def get_C3D_feature_extractor(summary=False):
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

def parse_vid(filename, remove_failures=False):
    vid = frontalizer.frontalize_video(filename)

    # This is a patch for cases in which frontalization does not work at all
    if (len(vid) == 0) and (remove_failures):
        return None
    elif (len(vid) == 0) and (not remove_failures):
        vid = np.array([np.zeros((128, 171, 3), dtype=np.float32)])

    n_frames = len(vid)
    n_clips = math.ceil(n_frames / clip_length)

    pad = np.zeros_like(vid[0], dtype=np.float32)

    n_missing_frames = (clip_length - (n_frames % clip_length)) % clip_length
    if n_missing_frames > 0:
        # Insert blank images
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

def clean_and_extract_C3D(videos, labels, remove_failures=False):
    """
    Extract C3D features from the input features and (optionally) remove samples in which there are
    no detected faces.
    """
    feature_extractor = get_C3D_feature_extractor()

    cln_features = []
    cln_labels = []
    for i, feat in enumerate(videos):
        parsed = parse_vid(feat, remove_failures=remove_failures)
        if parsed is not None:
            cln_features.append(parsed)
            cln_labels.append(labels[i])

    cln_features = [feature_extractor.predict(feat, batch_size=len(feat)) for feat in cln_features]
    cln_features = [[feat] if len(feat) == 112 else feat for feat in cln_features]

    return cln_features, cln_labels
