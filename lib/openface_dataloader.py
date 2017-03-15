# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - Dataloader for OpenFace FeatureExtraction's output file
===============================================================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

import os
import csv

def extract_columns(row, wanted_columns):
    """ 
    Extracts colums, ignoring those that do not appear in the row.
    
    Parameters
    ----------
    row : line of the feature's file
    wanted_columns : List with the desired column descriptors
        
    Returns
    -------
    Dict
        (column, value)
    """
    return dict((col, row[col]) for col in wanted_columns if col in row)

def extract_AUs(row):
    """ 
    Extracts the Action Units out of a single line from OpenFace feature's file.
    
    Parameters
    ----------
    row : line of the feature's file
        
    Returns
    -------
    Dict
        Action Units
    """
    return extract_columns(row, ['AU' + str(k) + '_c' for k in range(68)]
                                + ['AU' + str(k) + '_r' for k in range(68)])

def extract_2Dlandmarks(row):
    """ 
    Extracts the Landmarks out of a single line from OpenFace feature's file.
    
    Parameters
    ----------
    row : line of the feature's file
        
    Returns
    -------
    Dict
        Landmarks
    """
    return extract_columns(row, ['x_' + str(k) for k in range(68)]
                                + ['y_' + str(k) for k in range(68)])

if __name__ == "__main__":
    filename = os.getcwd() + "/datasets/ck+parsed/S005/001/openface_features.txt"

    with open(filename, 'r') as f:
        reader = csv.DictReader(f, skipinitialspace=True) # First line as 
                                                          # description of the columns
        for row in reader:
            print (extract_AUs(row))