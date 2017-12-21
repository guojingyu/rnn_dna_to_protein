"""
Implementation of RNN class and relevant algorithms

Author Jingyu Guo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def loss_per(self, y_hat, y):
    """
    :param y_hat: y_hat = decoder(encoder(x))
    y_hat is a softmax output.
    :param y: y is also an encoded prob distribution.
    :return: loss as defined by categorical entropy
    """
    return -1.0 * y * np.log(y_hat)


def total_cost(self, Y, X, Y_hat):
    """

    :param self:
    :param Y:
    :param X:
    :param Y_hat:
    :return:
    """
    N = len(Y)
    loss = 0.0
    for i in np.arange(N):
        precision_per_sample = Y_hat[np.arange(len(y[i])), y[i]]
        # Add to the loss based on how off we were
        loss += -1 * np.sum(np.log(precision_per_sample))
    return loss


class RNN(object):
    def __init__(self, input_dim=4, h_dim=100, o_dim=20, bptt_truncate=3, use_relu=False):
        """
        Initialize the RNN cell including all parameter matrics as U, W, and V
        with W being for previous hidden status h_t-1
        with U being for current input x_t
        with V being for output o_t
        The notations followed "Deep Learning" book by Goodfellow, etc.
        :param input_dim: the one-hot encoding of the vocabulary -- it might also be
        an embedded dense encoding dimension
        :param h_dim:
        :param bptt_truncate:
        """
        self.x_dim = input_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.bptt_truncate = bptt_truncate
        self.use_relu = use_relu
        # param init
        self.U = self.xavier_param_init(self.h_dim, self.x_dim, self.x_dim, self.use_relu)
        self.V = self.xavier_param_init(self.o_dim, self.h_dim, self.h_dim, self.use_relu)
        self.W = self.xavier_param_init(self.h_dim, self.h_dim, self.h_dim, self.use_relu)
        self.b = np.zeros([self.h_dim]).astype(np.float32)
        self.c = np.zeros([self.o_dim]).astype(np.float32)

    def xavier_param_init(self, r_dim, c_dim, prev_layer_dim, use_relu=False):
        """
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        :param r_dim:
        :param c_dim:
        :param prev_layer_dim:
        :param use_relu:
        :return:
        """
        if use_relu:
            d = 2.0
        else:
            d = 1.0
        return np.random.uniform(-np.sqrt(d/ prev_layer_dim),
                                 np.sqrt(d/prev_layer_dim), (r_dim, c_dim))

    def forward(self, X):
        """
        forward pass
        :param X: input -- 2d matrix with each row being an one hot encoding
        :return: tuple of size two (h_t, o_t)
        """
        pass
        #TODO implement
        return h_t, o_t

    def encoding_forward(self, X):
        """
        forward pass as encoding_forward
        :param X: input -- 2d matrix with each row being an one hot encoding
        :return: h_t
        """
        h_t, _ = self.forward(X)
        return h_t

    def decoding_forward(self, C, stop_vec):
        """

        :param C:
        :param stop_vec:
        :return:
        """
        pass


class Seq2Seq(object):
    def __init__(self):
        pass







