# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - Dataloader for Cohn-Kanade Database (CK and CK+) Emotion files
===============================================================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

import os
from subprocess import Popen, PIPE

def get_emotion(filename):
    """ 
    Extracts the emotion identifier from a CK+ Emotion label file.
    (0="Neutral" 1="Angry" 2="Contempt" 3="Disgust" 4="Fear" 5="Happy" 6="Sad"
    7="Surprise")

    Parameters
    ----------
    filename : emotion's file
    
    Returns
    -------
    Int
        Emotion identifier
    """
    text_file = open(filename,'rb')
    emotion = int(float(text_file.readline()))
    text_file.close()

    return emotion

def load_CK_emotions(root_dirname):
    """ 
    Loads all the emotion labels from CK+ database.

    Parameters
    ----------
    root_dirname : root directory of the CK+ dataset
    
    Returns
    -------
    Dict
        {Record identifier : Emotion identifier}
    """
    emotions={}

    emotion_dir = os.path.join(root_dirname, "Emotion")
    for dirname, _, file_list in os.walk(emotion_dir):
        for filename in file_list:
            record_id = filename[0:9]
            filename = os.path.join(dirname,filename)

            emotions.update({record_id : get_emotion(filename)})
    
    return emotions

def load_CK_videos(root_dirname):
    """ 
    Loads all the videos from CK+ database. Note: root_dirname must point at
    the parsed directory of the database (with processed videos from images).

    The reason the video filepath is returned instead of loading all videos is
    for performance issues.

    Parameters
    ----------
    root_dirname : root directory of the parsed CK+ dataset
    
    Returns
    -------
    Dict
        {Record identifier : video path}
    """
    videos = {}

    for dirname, _, file_list in os.walk(root_dirname):
        for filename in file_list:
            record_id = filename[0:9]
            filename = os.path.join(dirname, filename)

            videos.update({record_id : filename})

    return videos