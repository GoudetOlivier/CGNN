"""
Implementation of Losses
Author : Diviyan Kalainathan & Olivier Goudet
Date : 09/03/2017
"""

import tensorflow as tf
import numpy as np

bandwiths_gamma = [0.01, 0.1, 0.5,1, 2, 10, 100]

def MMD_loss_tf(xy_true, xy_pred, low_memory_version=False):

    N, _ = xy_pred.get_shape().as_list()

    if (low_memory_version == 1):

        loss = 0.0

        XX = tf.matmul(xy_pred, tf.transpose(xy_pred))
        X2 = tf.reduce_sum(xy_pred * xy_pred, 1, keep_dims=True)
        exponentXX = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)
        sXX = tf.constant(1.0 / N ** 2, shape=[N, 1])

        for i in range(len(bandwiths_gamma)):
            kernel_val = tf.exp(1.0 / bandwiths_gamma[i] * exponentXX)
            loss += tf.reduce_sum(sXX * kernel_val)

        YY = tf.matmul(xy_true, tf.transpose(xy_true))
        Y2 = tf.reduce_sum(xy_true * xy_true, 1, keep_dims=True)
        exponentYY = YY - 0.5 * Y2 - 0.5 * tf.transpose(Y2)
        sYY = tf.constant(1.0 / N ** 2, shape=[N, 1])

        for i in range(len(bandwiths_gamma)):
            kernel_val = tf.exp(1.0 / bandwiths_gamma[i] * exponentYY)
            loss += tf.reduce_sum(sYY * kernel_val)

        XY = tf.matmul(xy_pred, tf.transpose(xy_true))
        exponentXY = XY - 0.5 * X2 - 0.5 * tf.transpose(Y2)
        sXY = -tf.constant(2.0 / N ** 2, shape=[N, 1])

        for i in range(len(bandwiths_gamma)):
            kernel_val = tf.exp(bandwiths_gamma[i] * exponentXY)
            loss += tf.reduce_sum(sXY * kernel_val)

    else:

        X = tf.concat([xy_pred, xy_true], 0)
        XX = tf.matmul(X, tf.transpose(X))
        X2 = tf.reduce_sum(X * X, 1, keep_dims=True)
        exponent = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)

        s1 = tf.constant(1.0 / N, shape=[N, 1])
        s2 = -tf.constant(1.0 / N, shape=[N, 1])
        s = tf.concat([s1, s2], 0)
        S = tf.matmul(s, tf.transpose(s))

        loss = 0

        for i in range(len(bandwiths_gamma)):
            kernel_val = tf.exp(bandwiths_gamma[i] * exponent)
            loss += tf.reduce_sum(S * kernel_val)

    return tf.sqrt(loss)


def rp(k,s,d):

  return tf.transpose(tf.concat([tf.concat([si*tf.random_normal([k,d], mean=0, stddev=1) for si in s], axis = 0), 2*np.pi*tf.random_normal([k*len(s),1], mean=0, stddev=1)], axis = 1))

def f1(x,wz,N):

  ones = tf.ones((N, 1))
  x_ones = tf.concat([x, ones], axis = 1)
  mult = tf.matmul(x_ones,wz)

  return tf.cos(mult)

def Fourier_MMD_Loss_tf(xy_true, xy_pred,nb_vectors_approx_MMD):

  N, nDim = xy_pred.get_shape().as_list()

  wz = rp(nb_vectors_approx_MMD, bandwiths_gamma, nDim)

  e1 = tf.reduce_mean(f1(xy_true,wz,N), axis =0)
  e2 = tf.reduce_mean(f1(xy_pred,wz,N), axis =0)

  return tf.reduce_mean((e1 - e2)**2)


def MomentMatchingLoss_tf(xy_true, xy_pred, nb_moment = 1):
    """ k-moments loss, k being a parameter. These moments are raw moments and not normalized

    """
    loss = 0
    for i in range(1, nb_moment):
        mean_pred = tf.reduce_mean(xy_pred**i, 0)
        mean_true = tf.reduce_mean(xy_true**i, 0)
        loss += tf.sqrt(tf.reduce_sum((mean_true - mean_pred)**2))  # L2

    return loss
