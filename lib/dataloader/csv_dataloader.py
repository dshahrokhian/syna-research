# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - CSV Dataloader (to be extended by specific datasets)
===============================================================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

import csv

def open_and_extract(filename, function):
    """ 
    Opens a CSV file and extracts features based in the User-Defined Function 
    (UDF). For an example of this UDF, see (https://github.com/dshahrokhian/DeepMotion/blob/master/lib/openface_dataloader.py).
    
    Parameters
    ----------
    filename : csv file
    function : UDF
    
    Returns
    -------
    Dict
        {UDF key : UDF value}
    """
    features = {}
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, skipinitialspace=True) # Takes first line as 
                                                          # key of the columns
        for row in reader:
            features.update(function(row))
    
    return features