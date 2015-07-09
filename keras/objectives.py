from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from six.moves import range

epsilon = 1.0e-9

def bbox_mass(y_true, y_pred):
    m_true = y_true[:, 1:]
    y_true = y_true[:, 0]
    return T.sqr( y_pred.sum(axis=1) - y_true) + 0.1*T.sqr(y_pred * (1-m_true)).sum(axis=1)
        
def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean(axis=-1)

def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean(axis=-1)

def squared_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean(axis=-1)

def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean(axis=-1)

def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=-1, keepdims=True) 
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    return cce

def binary_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(y_pred, y_true)
    return bce

# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')

