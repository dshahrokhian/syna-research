#!/usr/bin/env python

# C3D imports
import matplotlib
matplotlib.use('Agg')
from keras.models import model_from_json, Sequential
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import deepmotion.c3d.c3d_model
import sys
import keras.backend as K

# DeepMotion imports
from keras.layers import Dense, LSTM, Activation, BatchNormalization, TimeDistributed, Dropout, Input

# Data Loader imports
from deepmotion.dataloader.ck_dataloader import load_CK_emotions, load_CK_videos
import deepmotion.dataloader.video_utils

dim_ordering = K.image_dim_ordering
print("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
        dim_ordering))
backend = dim_ordering

input_size = (112, 112)
clip_length = 16

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

def load_ck_data(videos_dir, emotion_dir):
    """ 
    Extracts video paths and CK+ Emotion labels, preserving the order 
    (e.g. x_train[0] corresponds to the same sample as y_train[0]).

    Parameters
    ----------
    videos_dir : root directory of the parsed CK+ dataset containing videos
    emotion_dir : root directory of the CK+ dataset
    
    Returns
    -------
    List, List, List, List
        train video-paths, test video-paths, 
        CK+ train emotion labels, CK+ test emotion labels
    """
    videos = load_CK_videos(videos_path)
    emotions = load_CK_emotions(emotions_path)

    # Convert to training structures
    features, labels = dicts2lists(videos, emotions)
    labels = to_categorical(labels)

    # Split into train and test sets
    train_size = int(len(features) * 0.67)
    test_size = len(features) - train_size
    x_train, x_test = np.array(features[0:train_size]), np.array(features[train_size:len(features)])
    y_train, y_test = np.array(labels[0:train_size]), np.array(labels[train_size:len(labels)])

    return x_train, x_test, y_train, y_test

def diagnose(data, verbose=True, label='input', plots=False, backend='tf'):
    # Convolution3D?
    if data.ndim > 2:
        if backend == 'th':
            data = np.transpose(data, (1, 2, 3, 0))
        #else:
        #    data = np.transpose(data, (0, 2, 1, 3))
        min_num_spatial_axes = 10
        max_outputs_to_show = 3
        ndim = data.ndim
        print("[Info] {}.ndim={}".format(label, ndim))
        print("[Info] {}.shape={}".format(label, data.shape))
        for d in range(ndim):
            num_this_dim = data.shape[d]
            if num_this_dim >= min_num_spatial_axes: # check for spatial axes
                # just first, center, last indices
                range_this_dim = [0, num_this_dim/2, num_this_dim - 1]
            else:
                # sweep all indices for non-spatial axes
                range_this_dim = range(num_this_dim)
            for i in range_this_dim:
                new_dim = tuple([d] + range(d) + range(d + 1, ndim))
                sliced = np.transpose(data, new_dim)[i, ...]
                print("[Info] {}, dim:{} {}-th slice: "
                      "(min, max, mean, std)=({}, {}, {}, {})".format(
                              label,
                              d, i,
                              np.min(sliced),
                              np.max(sliced),
                              np.mean(sliced),
                              np.std(sliced)))
        if plots:
            # assume (l, h, w, c)-shaped input
            if data.ndim != 4:
                print("[Error] data (shape={}) is not 4-dim. Check data".format(
                        data.shape))
                return
            l, h, w, c = data.shape
            if l >= min_num_spatial_axes or \
                h < min_num_spatial_axes or \
                w < min_num_spatial_axes:
                print("[Error] data (shape={}) does not look like in (l,h,w,c) "
                      "format. Do reshape/transpose.".format(data.shape))
                return
            nrows = int(np.ceil(np.sqrt(data.shape[0])))
            # BGR
            if c == 3:
                for i in range(l):
                    mng = plt.get_current_fig_manager()
                    mng.resize(*mng.window.maxsize())
                    plt.subplot(nrows, nrows, i + 1) # doh, one-based!
                    im = np.squeeze(data[i, ...]).astype(np.float32)
                    im = im[:, :, ::-1] # BGR to RGB
                    # force it to range [0,1]
                    im_min, im_max = im.min(), im.max()
                    if im_max > im_min:
                        im_std = (im - im_min) / (im_max - im_min)
                    else:
                        print("[Warning] image is constant!")
                        im_std = np.zeros_like(im)
                    plt.imshow(im_std)
                    plt.axis('off')
                    plt.title("{}: t={}".format(label, i))
                plt.show()
                #plt.waitforbuttonpress()
            else:
                for j in range(min(c, max_outputs_to_show)):
                    for i in range(l):
                        mng = plt.get_current_fig_manager()
                        mng.resize(*mng.window.maxsize())
                        plt.subplot(nrows, nrows, i + 1) # doh, one-based!
                        im = np.squeeze(data[i, ...]).astype(np.float32)
                        im = im[:, :, j]
                        # force it to range [0,1]
                        im_min, im_max = im.min(), im.max()
                        if im_max > im_min:
                            im_std = (im - im_min) / (im_max - im_min)
                        else:
                            print("[Warning] image is constant!")
                            im_std = np.zeros_like(im)
                        plt.imshow(im_std)
                        plt.axis('off')
                        plt.title("{}: o={}, t={}".format(label, j, i))
                    plt.show()
                    #plt.waitforbuttonpress()
    elif data.ndim == 1:
        print("[Info] {} (min, max, mean, std)=({}, {}, {}, {})".format(
                      label,
                      np.min(data),
                      np.max(data),
                      np.mean(data),
                      np.std(data)))
        print("[Info] data[:10]={}".format(data[:10]))

    return

def get_features_model(c3d_model):
    
    # Remove last 4 layers, which are for sports classification
    for _ in range(4):
        c3d_model.layers.pop()
        c3d_model.outputs = [c3d_model.layers[-1].output]
        c3d_model.layers[-1].outbound_nodes = []

    c3d_model.compile(optimizer='sgd', loss='mse')

    return c3d_model

def get_temporal_model(layers=[100]):

    temp_model = Sequential()
    
    temp_model.add(Input(batch_shape=(1, 1, 4096,), name='features'))
    temp_model.add(BatchNormalization(name='normalization'))
    temp_model.add(Dropout(p=.5))
    temp_model.add(LSTM(layers[-1], return_sequences=True, stateful=True, name='lsmt1'))
    temp_model.add(Dropout(p=.5))
    temp_model.add(TimeDistributed(Dense(8, activation='softmax'), name='fc'))

    temp_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return temp_model

def get_vid(filename):
    print("[Info] Loading a sample video...")
    cap = cv2.VideoCapture(filename)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    vid = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        vid.append(cv2.resize(img, (171, 128)))
    vid = np.array(vid, dtype=np.float32)

    #plt.imshow(vid[2000]/256)
    #plt.show()

    # sample clip_length
    #start_frame = 100
    start_frame = 2000

    X = vid[start_frame:(start_frame + clip_length), :, :, :]
    #diagnose(X, verbose=True, label='X (clip_length-frame clip)', plots=show_images)

    # subtract mean
    mean_cube = np.load('models/train01_16_128_171_mean.npy')
    mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
    #diagnose(mean_cube, verbose=True, label='Mean cube', plots=show_images)
    X -= mean_cube
    #diagnose(X, verbose=True, label='Mean-subtracted X', plots=show_images)

    # center crop
    X = X[:, 8:120, 30:142, :] # (l, h, w, c)
    #diagnose(X, verbose=True, label='Center-cropped X', plots=show_images)

def main():
    show_images = False
    diagnose_plots = False
    model_dir = os.path.join(os.path.dirname(__file__), 'deepmotion/c3d/models')
    global backend

    epochs = 6
    batch_size = 1

    # override backend if provided as an input arg
    if len(sys.argv) > 1:
        if 'tf' in sys.argv[1].lower():
            backend = 'tf'
        else:
            backend = 'th'
    print("[Info] Using backend={}".format(backend))

    if backend == 'th':
        model_weight_filename = os.path.join(model_dir, 'sports1M_weights_th.h5')
        model_json_filename = os.path.join(model_dir, 'sports1M_weights_th.json')
    else:
        model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
        model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

    print("[Info] Reading model architecture...")
    model = model_from_json(open(model_json_filename, 'r').read())

    print("[Info] Loading model weights...")
    model.load_weights(model_weight_filename)
    print("[Info] Loading model weights -- DONE!")

    print("[Info] Loading labels...")
    with open('datasets/labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))

    # DeepMotion
    features_model = get_features_model(model)
    temporal_model = get_temporal_model()

    # Load videos and emotions
    videos_path = os.path.join(os.path.dirname(__file__),  "datasets/ck+videos")
    emotions_path = os.path.join(os.path.dirname(__file__), "datasets/ck+")
    
    x_train, x_test, y_train, y_test = load_ck_data(videos_path, emotions_path)
    for _ in range(epochs):
        for X, Y in zip(x_train, y_train):
            model.train_on_batch(np.array([X]), np.array([Y]))
        
        # Final evaluation of the model
        acc = 0
        samples = 0
        for X, Y in zip(x_test, y_test):
            acc += model.test_on_batch(np.array([X]), np.array([Y]))[1]
            samples += 1
        print("Accuracy: %.2f%%" % (acc/samples*100))    

    # get activations for intermediate layers if needed
    inspect_layers = [
    #    'fc6',
    #    'fc7',
        ]
    for layer in inspect_layers:
        int_model = c3d_model.get_int_model(model=model, layer=layer, backend=backend)
        int_output = int_model.predict_on_batch(np.array([X]))
        int_output = int_output[0, ...]
        print("[Debug] at layer={}: output.shape={}".format(layer, int_output.shape))
        diagnose(int_output,
                 verbose=True,
                 label='{} activation'.format(layer),
                 plots=diagnose_plots,
                 backend=backend)

if __name__ == '__main__':
    main()
