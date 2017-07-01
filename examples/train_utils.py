# -*- coding: utf-8 -*-
"""
===============================================================================
DeepMotion - Training Utils
===============================================================================
"""
# Author: Daniyal Shahrokhian <daniyal@kth.se>

import os
import dlib
import scipy.misc as sm
import matplotlib.pyplot as plt
import numpy as np
from deepmotion.frontalization import facefrontal
from sklearn import preprocessing

class face_frontalizer():
    """
    This is an adaptation from https://github.com/ChrisYang/facefrontalisation. 
    I appologize if the code is not very clean, but there are certain 
    implementation aspects that were not explained neither in his python 
    conversion nor in the original matlab code.
    """
    def __init__(self):
        face_model_path = os.path.join(os.path.dirname(__file__),  '../data/frontalization/face_shape.dat')
        ref3d_path = os.path.join(os.path.dirname(__file__),  '../data/frontalization/ref3d.pkl')
        
        self.face_model = dlib.shape_predictor(face_model_path)
        self.face_detector = dlib.get_frontal_face_detector()
        self.frontalizer = facefrontal.frontalizer(ref3d_path)

    def frontalize(self, img):
        """ 
        Given a path to an image, it returns the same image frontalized.

        Parameters
        ----------
        filename : image file
        
        Returns
        -------
        PIL.Image
            Frontalized image
        """

        # Ask the detector to find the bounding boxes of each face. The ':,:,:3' 
        # in the first argument avoids the last dimension in RGBA formats, 
        # which dlib doesn't accept. The 1 in the second argument indicates that 
        # we should upsample the image 1 time. This will make everything bigger 
        # and allow us to detect more faces.
        detected_faces = self.face_detector(img[:,:,:3], 1)
        
        frontalized_faces = []
        for face in detected_faces:
            shape = self.face_model(img,face)
            p2d = np.asarray([(shape.part(n).x, shape.part(n).y,) for n in range(shape.num_parts)], np.float32)

            rawfront, symfront = self.frontalizer.frontalization(img,face,p2d)
            frontalized_faces.append(sm.toimage(np.round(symfront).astype(np.uint8)))
        
        return frontalized_faces

def normalize(X, axis=0):
    """ 
    Given a list of samples, centers to the mean and component wise scales to
    unit variance.

    Parameters
    ----------
    X : list of samples
    axis : dimension to normalize

    Returns
    -------
    List
        Normalized list of samples
    """
    # Flatten the array and apply scaling
    flat_X = np.concatenate(X, axis=axis)
    flat_X = preprocessing.scale(flat_X)

    # Convert to original shape
    i, j = 0, 0
    while j < len(flat_X):
        X[i] = flat_X[j:j+len(X[i])]

        j += len(X[i])
        i += 1
    
    return X

def evaluate(model, features, labels):
    scores = []
    for X, Y in zip(features, labels):
        scores.append(model.test_on_batch(np.array([X]), np.array([Y])))
    
    return scores

if __name__ == "__main__":
    img = plt.imread('/home/dani/Downloads/pic3.jpg')
    frontalizer = face_frontalizer()
    frontalizer.frontalize(img)[0].show()