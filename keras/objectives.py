from __future__ import absolute_import
import numpy as np
from . import backend as K

if theano.config.floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7

def bce(y_true, y_pred):
    y_pred = T.clip(y_pred[:, 0], epsilon, 1.0 - epsilon)
    return T.nnet.binary_crossentropy(y_pred, y_true[:, 0]).mean(axis=(-2,-1))


def masked_xy_loss(m_true, y_true, y_pred, l_func):
    # m_true is (BATCH_SIZE, H, W)
    # y_pred and y_true are: (BATCH_SIZE, k, H, W)
    return (m_true * l_func(y_pred - y_true).sum(axis=-3)).sum(axis=(-2,-1)) \
        / (epsilon + m_true.sum(axis=(-2,-1)))


def xy_l1(y_true, y_pred):
    return masked_xy_loss(y_true[:, 0] * y_true[:, 7], y_true[:, 5:7], y_pred[:, 5:7], T.abs_)


def xy_l2(y_true, y_pred):
    return masked_xy_loss(y_true[:, 0] * y_true[:, 7], y_true[:, 5:7], y_pred[:, 5:7], T.sqr)


def jaccard(y_true, y_pred):
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
    return jaccard


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
    return T.sqr(y_pred - y_true).mean(axis=-1)


def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean(axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return K.categorical_crossentropy(y_pred, y_true)


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)


# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity

from .utils.generic_utils import get_from_module
def get(identifiers_str):
    identifiers_lst = identifiers_str.split(',')
    use_sigmoid = 'sigmoid' in identifiers_lst
    funcs = [get_from_module(identifier, globals(), 'objective') for identifier in identifiers_lst \
                                                                 if identifier <> 'sigmoid']
    def sum_of_funcs(y_true, y_pred):
        if use_sigmoid:
            y_pred = T.concatenate([T.nnet.sigmoid(y_pred[:, 0:1]), y_pred[:, 1:]], axis=1)
        res = 0
        for f in funcs:
            res += f(y_true, y_pred)
        return res
    return sum_of_funcs



