# -*- coding: utf-8 -*-
"""
===============================================================================
Syna - Experimenting with C3D features
===============================================================================
"""
__author__ = "Daniyal Shahrokhian <daniyal@kth.se>"

from bayes_opt import BayesianOptimization

import c3d_utils
import io_utils
import train_utils
from deepmotion.dataloader.ck_dataloader import (load_CK_emotions,
                                                 load_CK_videos)
from deepmotion.deepmotion_model import get_temporal_model


def dicts2lists(dict_videos, dict_emotions):
    """
    Converts the dictionaries of the dataloaders into lists, containing only
    records with identifiers present in both dictionaries, and ordered by
    record identifiers.

    Parameters
    ----------
    dict_videos : {Record identifier : video path}
    dict_emotions : {Record identifier : Emotion identifier}

    Returns
    -------
    List, List
        [videp paths], [emotion identifiers]
    """
    l_videos = []
    l_emotions = []

    for record_id, video_path in dict_videos.items():
        if record_id in dict_emotions:
            l_videos.append(video_path)
            l_emotions.append(dict_emotions[record_id])

    return l_videos, l_emotions

def load_ck_data(videos_path, emotions_path):
    """
    Extracts video paths and CK+ Emotion labels, preserving the order
    (e.g. x_train[0] corresponds to the same sample as y_train[0]).

    Parameters
    ----------
    videos_path : root directory of the parsed CK+ dataset containing videos
    emotion_path : root directory of the CK+ dataset

    Returns
    -------
    List, List
        video-paths, CK+ emotion labels
    """
    videos = load_CK_videos(videos_path)
    emotions = load_CK_emotions(emotions_path)

    return dicts2lists(videos, emotions)

def main():
    # Load videos and emotions
    videos_path = os.path.join(os.path.dirname(__file__), "../datasets/ck+videos")
    emotions_path = os.path.join(os.path.dirname(__file__), "../datasets/ck+")
    videos, labels = load_ck_data(videos_path, emotions_path)

    # Extract features
    features, labels = c3d_utils.clean_and_extract_C3D(videos,labels)

    # Bayesian Hyperparameter Optimization
    evaluator = train_utils.KFoldEvaluator(get_temporal_model, features, labels)
    hyper_opt = BayesianOptimization(evaluator.evaluate, {'neurons': (40, 200),
                                                          'epochs': (5, 40),
                                                          'lr': (0.0005, 0.005),
                                                          'lr_decay': (0.0, 1e-4),
                                                          'batch_size': (1, 1)
                                                         })
    hyper_opt.maximize()
    optimal = hyper_opt.res['max']

    print("Best hyperparameter settings: " + str(optimal))
    io_utils.kfold_report_metrics(get_temporal_model, optimal['max_params'], features, labels)

if __name__ == '__main__':
    main()
