# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - LSTM
===============================================================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

from keras.layers import (LSTM, Activation, BatchNormalization, Dense, Dropout, Input)
from keras.models import Sequential, Model
from keras.optimizers import Adam


def get_model(summary=False, layers=[100], lr=0.001, lr_decay=0.0, input_shape=(10, 64)):
    """
    Returns the Keras model of the network.

    Parameters
    ----------
    summary : print model summary
    layers : list with the number of LSTM units per layer
    lr : learning rate
    lr_decay : learning rate decay
    input_shape : input shape to the network

    Returns
    -------
    Sequential
        Keras model
    """
    model = Sequential()

    for layer in layers[:-1]:
        model.add(LSTM(layer, input_shape=input_shape, return_sequences=True))
        model.add(Activation('relu'))
    model.add(LSTM(layers[-1], input_shape=input_shape))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    adam_opt = Adam(lr=lr, decay=lr_decay)
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

    if summary:
        print(model.summary())

    return model

def get_temporal_model(summary=False, layers=[100], lr=0.001, lr_decay=0.0,
                       input_shape=(None, 4096, )):
    """
    Returns the Keras model of the network.

    Parameters
    ----------
    summary : print model summary
    layers : list with the number of LSTM units per layer
    lr : learning rate
    lr_decay : learning rate decay
    input_shape : input shape to the network

    Returns
    -------
    Sequential
        Keras model
    """
    input_features = Input(shape=input_shape, name='features')
    #input_normalized = BatchNormalization(name='normalization')(input_features)
    #input_dropout = Dropout(rate=0.5)(input_normalized)
    input_dropout = Dropout(rate=0.5)(input_features)
    lstm = LSTM(layers[-1], name='lsmt1')(input_dropout)
    output_dropout = Dropout(rate=0.5)(lstm)
    #output = TimeDistributed(Dense(8, activation='softmax'), name='fc')(output_dropout)
    output = Dense(8, activation='softmax', name='fc')(output_dropout)

    temp_model = Model(inputs=input_features, outputs=output)

    adam_opt = Adam(lr=lr, decay=lr_decay)
    temp_model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

    if summary:
        print(model.summary())

    return temp_model
