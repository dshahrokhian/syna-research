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
    return dict((col, row[col]) for col in wanted_columns)

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
    return extract_columns(row, ['AU01_r',
                                'AU02_r',
                                'AU04_r',
                                'AU05_r',
                                'AU06_r',
                                'AU07_r',
                                'AU09_r',
                                'AU10_r',
                                'AU12_r',
                                'AU14_r',
                                'AU15_r',
                                'AU17_r',
                                'AU20_r',
                                'AU23_r',
                                'AU25_r',
                                'AU26_r',
                                'AU45_r',
                                'AU01_c',
                                'AU02_c',
                                'AU04_c',
                                'AU05_c',
                                'AU06_c',
                                'AU07_c',
                                'AU09_c',
                                'AU10_c',
                                'AU12_c',
                                'AU14_c',
                                'AU15_c',
                                'AU17_c',
                                'AU20_c',
                                'AU23_c',
                                'AU25_c',
                                'AU26_c',
                                'AU28_c',
                                'AU45_c'])

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
    raise NotImplementedError

if __name__ == "__main__":
    filename = os.getcwd() + "/datasets/ck+parsed/S005/001/openface_features.txt"
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, skipinitialspace=True) # First line as 
                                                          # description of the columns
        for row in reader:
            print (extract_AUs(row))