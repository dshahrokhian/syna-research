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
                                ['AU' + "{0:02d}".format(k) + '_c' for k in range(46)]
                                + ['AU' + "{0:02d}".format(k) + '_r' for k in range(46)])}

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

def load_AU_activations(root_dirname):
    """ 
    Loads all the Action Units activations from the parsed CK+ database. See
    (https://github.com/dshahrokhian/DeepMotion/blob/master/datasets/parse_dataset.sh).

    Parameters
    ----------
    root_dirname : root directory of the parsed CK+ dataset
    
    Returns
    -------
    Dict
        {Record identifier : {timestamp : {AU code : AU activation value}}}
    """
    activations={}

    for dirname, _, file_list in os.walk(root_dirname):
        for filename in file_list:
            if filename.endswith(".txt"):
                record_id = filename[0:8]
                filename = os.path.join(dirname,filename)

                activations.update({record_id : get_AU_activations(filename)})
    
    return activations