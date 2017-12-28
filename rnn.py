"""
Implementation of RNN class and relevant algorithms

Author Jingyu Guo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def cross_entropy_loss(self, Y_hat, Y):
    """
    The loss function here is defined as an average over all amino acid
    cross entropy loss (regardless which samples they are from) .
    It is good to concatenate all samples (amino acid string vectors) in a
    batch together.
    :param Y_hat: Y_hat is a collection of y_hat, such as y_hat = decoder(
    encoder(x)). y_hat is a softmax output vector for one amino acid.
    :param Y: Y is a collection of Y. y is also an encoded one hot vector for
    one amino acid.
    :return: loss as defined by categorical entropy
    """
    loss = 0.0
    for y_hat, y in zip(Y_hat, Y):
        loss += -1.0 * np.sum(np.multiply(y, np.log(y_hat)))
    return loss/len(Y_hat)

def softmax(x):
    """
    Not completely following the mathematic defintion of softmax.
    As suggested by Stanford cs231n about softmax,
    to shift the values of x so that the highest number is 0.
    This will lead to better numeric stability.
    :param x: a distribution of numbers as a numpy array
    :return: normalized vector as a numpy array
    """
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


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
        self.time_step = 0
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
        :return: tuple of size two (H, S, O)
        """
        self.time_step = len(X)
        # time_step + 1 to add an initial hidden state for the neuron
        H = np.zeros((self.time_step + 1, self.h_dim))
        O = np.zeros((self.time_step, self.o_dim))
        for t in range(self.time_step):
            H[t] = np.tanh(self.U[:, X[t]] + self.W.dot(H[t - 1]))
            O[t] = softmax(self.V.dot(H[t]))
        return H, O

    def encoding_forward(self, X):
        """
        forward pass as encoding_forward
        :param X: input -- 2d matrix with each row being an one hot encoding
        :return: h_t
        """
        H, O = self.forward(X)
        return H, O[len(X)] # discarded other outputs but the last one

    def decoding_forward(self, C, W, stop_vec):
        """
        forward pass as decoding steps
        :param C: the input hidden status or context (thus C)
        :param W: the last output from encoder
        :param stop_vec: use for ending the pass
        :param stop_vec:
        :return:
        """
        H = np.zeros((1, self.h_dim))
        O = np.zeros((time_step, self.h_dim))
        while :
            self.time_step += 1
            H[t] = np.tanh(self.U[:, X[t]] + self.W.dot(H[t - 1]))
            O[t] = softmax(self.V.dot(H[t]))
        return H, O

    def backprop_through_time(self, error, k1):
        """
        A truncated back prop through timestep method for rnn training.

        Truncated BPTT is a closely related method. It processes the sequence
        one timestep at a time, and every k1 timesteps, it runs BPTT for k2
        timesteps, so a parameter update can be cheap if k2 is small.
        Consequently, its hidden states have been exposed to many timesteps
        and so may contain useful information about the far past, which
        would be opportunistically exploited.
        By Ilya Sutskever, Training Recurrent Neural Networks, Dissertation
        2013

        Present a sequence of k1 timesteps of input and output pairs to the network.
        Unroll the network then calculate and accumulate errors across k2 timesteps.
        Roll-up the network and update weights.
        Repeat
        https://machinelearningmastery.com/gentle-introduction-backpropagation-time/
        """
        # in case k1 (which would be input sequence length) is smaller than
        # predefined self.bptt_truncate
        if k1 < self.bptt_truncate:
            k2 = k1
        else:
            k2 = self.bptt_truncate

        for i in range(k1)# k1
        for i in range(bptt_truncate): # k2



class Seq2Seq(object):
    def __init__(self):
        pass







