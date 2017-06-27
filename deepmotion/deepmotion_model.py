# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - LSTM Neural Network integration for OpenFace
===============================================================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import Adam

def get_model(summary=False, layers=[100], lr=0.001, lr_decay=0.0, input_shape=(10, 64)):
    """ 
    Returns the Keras model of the network.
    
    Parameters
    ----------
    summary : print model summary
    layers : list with the number of LSTM units per layer
    lr : learning rate
    lr : learning rate decay
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

if __name__ == '__main__':
    model = get_model(summary=True)
