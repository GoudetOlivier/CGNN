"""
Implementation of Losses
Author : Diviyan Kalainathan & Olivier Goudet
Date : 09/03/2017
"""

import tensorflow as tf
import numpy as np

bandwiths_gamma = [0.005, 0.05, 0.25, 0.5, 1, 5, 50]

def MMD_loss_tf(xy_true, xy_pred):

    N, _ = xy_pred.get_shape().as_list()

    X = tf.concat([xy_pred, xy_true], 0)
    XX = tf.matmul(X, tf.transpose(X))
    X2 = tf.reduce_sum(X * X, 1, keep_dims=True)
    exponent = -2*XX + X2 + tf.transpose(X2)

    s1 = tf.constant(1.0 / N, shape=[N, 1])
    s2 = -tf.constant(1.0 / N, shape=[N, 1])
    s = tf.concat([s1, s2], 0)
    S = tf.matmul(s, tf.transpose(s))

    loss = 0

    for i in range(len(bandwiths_gamma)):
        kernel_val = tf.exp(-bandwiths_gamma[i] * exponent)
        loss += tf.reduce_sum(S * kernel_val)

    return loss


def rp(k,s,d):

  return tf.transpose(tf.concat([tf.concat([2*si*tf.random_normal([k,d], mean=0, stddev=1) for si in s], axis = 0), tf.random_uniform([k*len(s),1], minval=0, maxval=2*np.pi)], axis = 1))

def f1(x,wz,N):

  ones = tf.ones((N, 1))
  x_ones = tf.concat([x, ones], axis = 1)
  mult = tf.matmul(x_ones,wz)

  return tf.cos(mult)

def Fourier_MMD_Loss_tf(xy_true, xy_pred,nb_vectors_approx_MMD):

  N, nDim = xy_pred.get_shape().as_list()

  wz = rp(nb_vectors_approx_MMD, bandwiths_gamma, nDim)

  e1 = tf.sqrt(2/nb_vectors_approx_MMD)*tf.reduce_mean(f1(xy_true, wz, N), axis=0)
  e2 = tf.sqrt(2/nb_vectors_approx_MMD)*tf.reduce_mean(f1(xy_pred, wz, N), axis=0)

  return tf.reduce_sum((e1 - e2) ** 2)
    



def MomentMatchingLoss_tf(xy_true, xy_pred, nb_moment = 1):
    """ k-moments loss, k being a parameter. These moments are raw moments and not normalized

    """
    loss = 0
    for i in range(1, nb_moment):
        mean_pred = tf.reduce_mean(xy_pred**i, 0)
        mean_true = tf.reduce_mean(xy_true**i, 0)
        loss += tf.sqrt(tf.reduce_sum((mean_true - mean_pred)**2))  # L2

    return loss
