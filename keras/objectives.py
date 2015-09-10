from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from six.moves import range

if theano.config.floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7

def bbox_mse(y_true, y_pred):
    # y_true and y_pred are both (BATCH_SIZE, 5, 36, 71).
    m_true = y_true[:, 0]
    m_pred = y_pred[:, 0]
    m_pred = T.clip(m_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(m_pred, m_true).sum(axis=(-2,-1))

    b_true = y_true[:, 1:]
    b_pred = y_pred[:, 1:]
    mse = T.sqr(y_true[:, :1] * (b_pred[:, 1:] - b_true[:, 1:])).sum(axis=(-3,-2,-1))
    return bce + mse

def bbox_jaccard(y_true, y_pred):
    # y_true and y_pred are both (BATCH_SIZE, 5, 36, 71).
    m_true = y_true[:, 0]
    m_pred = y_pred[:, 0]
    m_pred = T.clip(m_pred, epsilon, 1.0 - epsilon)
    bce = T.nnet.binary_crossentropy(m_pred, m_true).mean(axis=(-2,-1))

    # Find intersection rectangle.
    top = T.maximum(y_true[:,1], y_pred[:,1])
    bottom = T.minimum(y_true[:,2], y_pred[:,2])
    left = T.maximum(y_true[:,3], y_pred[:,3])
    right = T.minimum(y_true[:,4], y_pred[:,4])
    intersection = T.maximum(bottom - top, 0) * T.maximum(right - left, 0)
    area_true = (y_true[:,2] - y_true[:,1]) * (y_true[:,4] - y_true[:,3])
    area_pred = T.maximum(y_pred[:,2] - y_pred[:,1], 0) * T.maximum(y_pred[:,4] - y_pred[:,3], 0)
    union = T.maximum(area_true + area_pred - intersection, epsilon)
    jaccard = (y_true[:,0] * (1 - intersection / union)).sum(axis=(-2,-1)) \
        / (epsilon + y_true[:,0].sum(axis=(-2,-1)))

#    if 0:
#        d_pred = y_pred[:, 5]                      # d_pred is (BATCH_SIZE, H, W).
#        d_pred = T.maximum(0, d_pred)
#        n_true = y_true[:, 5].max(axis=(-2,-1))  # n_true is now (BATCH_SIZE,)
#        d_true = y_true[:, 5] > 0                # d_true is (BATCH_SIZE, H, W)
#        mass = T.abs_(d_pred.sum(axis=(-2,-1)) - n_true) + T.abs_(d_pred * (1 - d_true)).sum(axis=(-2,-1))

    return  bce + jaccard #+ mass

def bbox_mass(y_true, y_pred):
    # y_true is (BATCH_SIZE, 2, H, W).
    # y_pred is (BATCH_SIZE, 6, H, W).
    d_pred = (y_pred[:, 5] + T.abs_(y_pred[:, 5])) / 2.0 # d_pred is (BATCH_SIZE, H, W).
    n_true = y_true[:, 0].max(axis=(-2,-1))  # n_true is now (BATCH_SIZE,)
    d_true = y_true[:, 0] > 0                # d_true is (BATCH_SIZE, H, W)
    mass = T.abs_(d_pred.sum(axis=(-2,-1)) - n_true) + T.abs_(d_pred * (1 - d_true)).sum(axis=(-2,-1))
    return mass

def mse_plus_mae(y_true, y_pred):
   return mean_squared_error(y_true, y_pred) + mean_absolute_error(y_true, y_pred)

def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean(axis=(1,2,3))


def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean(axis=(1,2,3))


def mean_absolute_percentage_error(y_true, y_pred):
    return T.abs_((y_true - y_pred) / T.clip(T.abs_(y_true), epsilon, np.inf)).mean(axis=-1) * 100.


def mean_squared_logarithmic_error(y_true, y_pred):
    return T.sqr(T.log(T.clip(y_pred, epsilon, np.inf) + 1.) - T.log(T.clip(y_true, epsilon, np.inf) + 1.)).mean(axis=-1)


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
    bce = T.nnet.binary_crossentropy(y_pred, y_true).mean(axis=-1)
    return bce


def poisson_loss(y_true, y_pred):
    return T.mean(y_pred - y_true * T.log(y_pred), axis=-1)

# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
