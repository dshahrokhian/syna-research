# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - Dataloader for OpenFace FeatureExtraction's output file
===============================================================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

import os
import csv

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
        (column, value)
    """
    return {col : row[col] for col in wanted_columns if col in row}

def _extract_AUs(row):
    """ 
    Extracts the Action Units out of a single line from OpenFace feature's file.
    
    Parameters
    ----------
    row : line of the feature's file
        
    Returns
    -------
    Dict
        (timestamp, Dict(AU code, AU value))
    """
    return {row['timestamp'] : _extract_columns(row, ['AU' + str(k) + '_c' for k in range(46)]
                                + ['AU' + str(k) + '_r' for k in range(46)])}

def _extract_2Dlandmarks(row):
    """ 
    Extracts the Landmarks out of a single line from OpenFace feature's file.
    
    Parameters
    ----------
    row : line of the feature's file
        
    Returns
    -------
    Dict
        (timestamp, Dict(Landmark code, Landmark value))
    """
    return {row['timestamp'] : _extract_columns(row, ['x_' + str(k) for k in range(68)]
                                + ['y_' + str(k) for k in range(68)])}

def _open_and_extract(filename, function):
    features = {}
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, skipinitialspace=True) # Takes first line as 
                                                          # key of the columns
        for row in reader:
            features.update(function(row))

    return features

def get_2Dlandmarks(filename):
    _open_and_extract(filename, _extract_2Dlandmarks)

def get_AUs(filename):
    _open_and_extract(filename, _extract_AUs)