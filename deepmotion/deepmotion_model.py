# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - LSTM Neural Network integration for OpenFace
===============================================================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation

def get_model(summary=False, layers=[100], input_shape=(10, 64)):
    """ 
    Returns the Keras model of the network.
    
    Parameters
    ----------
    summary : print model summary
    layers : list with the number of LSTM units per layer 
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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if summary:
        print(model.summary())

    return model

if __name__ == '__main__':
    model = get_model(summary=True)
