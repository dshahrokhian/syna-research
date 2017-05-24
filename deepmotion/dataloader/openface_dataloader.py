# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - Dataloader for OpenFace FeatureExtraction's output file
===============================================================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

import os
from .csv_dataloader import open_and_extract

def _extract_columns(row, wanted_columns):
    """ 
    Extracts columns, ignoring those that do not appear in the row.
    
    Parameters
    ----------
    row : line of the feature's file
    wanted_columns : list with the desired column descriptors
        
    Returns
    -------
    Dict
        {column : value}
    """
    return {col : row[col] for col in wanted_columns if col in row}

def _extract_AU_activations(row):
    """ 
    Extracts the Action Units Activations (Boolean 0 or 1) out of a single line
    from OpenFace feature's file.
    
    Parameters
    ----------
    row : line of the feature's file
        
    Returns
    -------
    Dict
        {timestamp : {AU code : AU value}}
    """
    return {row['timestamp'] : _extract_columns(row, 
                                ['AU' + "{0:02d}".format(k) + '_c' for k in range(46)])}

def _extract_AUs(row):
    """ 
    Extracts the Action Units out of a single line from OpenFace feature's file.
    
    Parameters
    ----------
    row : line of the feature's file
        
    Returns
    -------
    Dict
        {timestamp : {AU code : AU value}}
    """
    return {row['timestamp'] : _extract_columns(row, 
                                ['AU' + "{0:02d}".format(k) + '_r' for k in range(46)])}

def _extract_2Dlandmarks(row):
    """ 
    Extracts the Landmarks out of a single line from OpenFace feature's file.
    
    Parameters
    ----------
    row : line of the feature's file
        
    Returns
    -------
    Dict
        {timestamp : {Landmark code : Landmark value}}
    """
    return {row['timestamp'] : _extract_columns(row, ['x_' + str(k) for k in range(68)]
                                + ['y_' + str(k) for k in range(68)])}

def get_2Dlandmarks(filename):
    """ 
    Extracts the Landmarks from an OpenFace feature's file.
    
    Parameters
    ----------
    filename : feature's file
        
    Returns
    -------
    Dict
        {timestamp : {Landmark code : Landmark value}}
    """
    return open_and_extract(filename, _extract_2Dlandmarks)

def get_normalized_landmarks(filename):
    """
    Extracts the Landmarks from an OpenFace feature's file, and normalizes them
    in the range [-1,1] in both x and y axes.
    
    Parameters
    ----------
    filename : feature's file
        
    Returns
    -------
    Dict
        {timestamp : {Landmark code : Landmark value}}
    """
    dict_landmarks = get_2Dlandmarks(filename)

    for timestamp, landmarks in dict_landmarks.items():
        dict_landmarks.update({timestamp : _normalize(landmarks)})
    
    return dict_landmarks

def get_AUs(filename):
    """ 
    Extracts the Action Units out of an OpenFace feature's file.
    
    Parameters
    ----------
    filename : feature's file
        
    Returns
    -------
    Dict
        {timestamp : {AU code : AU value}}
    """
    return open_and_extract(filename, _extract_AUs)

def get_AU_activations(filename):
    """ 
    Extracts the Action Units Activations (Boolean 0 or 1) out of an OpenFace 
    feature's file.
    
    Parameters
    ----------
    filename : feature's file
        
    Returns
    -------
    Dict
        {timestamp : {AU code : AU activation value}}
    """
    return open_and_extract(filename, _extract_AU_activations)

def load_OpenFace_features(root_dirname, features='AUs'):
    """ 
    Loads all the features from the parsed CK+ or AFEW datasets. See
    (https://github.com/dshahrokhian/DeepMotion/blob/master/datasets/parse_dataset.sh)
    if you need to understand the parsed data directory structure.

    Parameters
    ----------
    root_dirname : root directory of the parsed CK+ dataset
    features : which features to load {AUs, AU_activations, 2Dlandmarks}

    Returns
    -------
    Dict
        {Record identifier : {timestamp : {feature code : feature value}}}
    """
    output={}

    for dirname, _, file_list in os.walk(root_dirname):
        for filename in file_list:
            if filename.endswith(".txt"):
                record_id = filename[0:9]
                filename = os.path.join(dirname,filename)

                output.update({record_id : globals()['get_' + features](filename)})
    
    return output