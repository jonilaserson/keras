from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from six.moves import range

epsilon = 1.0e-9
   
def bbox_mass(y_true, y_pred, weight=None):
    m_true = y_true[:, 1:]
    y_true = y_true[:, 0]
    if weight is None:
        return T.sqr( (y_pred * (1-m_true)).sum(axis=1)).mean() + \
            T.sqr( y_pred.sum(axis=1) - y_true).mean()
    else:
        w = weight.reshape((weight.shape[0], 1)) 
        return (w * T.sqr( (y_pred * (1-m_true)).sum(axis=1))).mean() + \
            (w * T.sqr(y_pred.sum(axis=1) - y_true)).mean()
        
def mean_squared_error(y_true, y_pred, weight=None):
    if weight is not None:
        return T.sqr(weight.reshape((weight.shape[0], 1)) * (y_pred - y_true)).mean()
    else:
        return T.sqr(y_pred - y_true).mean()

def mean_absolute_error(y_true, y_pred, weight=None):
    if weight is not None:
        return T.abs_(weight.reshape((weight.shape[0], 1)) * (y_pred - y_true)).mean()
    else:
        return T.abs_(y_pred - y_true).mean()

def squared_hinge(y_true, y_pred, weight=None):
    if weight is not None:
        weight = weight.reshape((weight.shape[0], 1))
        return T.sqr(weight * T.maximum(1. - (y_true * y_pred), 0.)).mean()
    else:
        return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean()

def hinge(y_true, y_pred, weight=None):
    if weight is not None:
        weight = weight.reshape((weight.shape[0], 1))
        return (weight * T.maximum(1. - (y_true * y_pred), 0.)).mean()
    else:
        return T.maximum(1. - y_true * y_pred, 0.).mean()

def categorical_crossentropy(y_true, y_pred, weight=None):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=1, keepdims=True) 
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    if weight is not None:
        # return avg. of scaled cat. crossentropy
        return (weight * cce).mean()
    else:
        return cce.mean()

def binary_crossentropy(y_true, y_pred, weight=None):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(y_pred, y_true)
    if weight is not None:
        return (weight.reshape((weight.shape[0], 1)) * bce).mean()
    else:
        return bce.mean()

# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')