"""
Implementation of RNN class and relevant algorithms

Author Jingyu Guo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from data_processing import encoding, decoding, reverse_seq
from vocabulary import TRUE_DNA_CODON_TABLE, NUCLEOTIDE, AMINO_ACID, \
    STOP_CODONS, END_OF_SENTENCE

def padding_vector(v1, v2):
    """
    padding the 2d shorter matrix by the end with zeros to be taller
    :param v1: numpy 2d array
    :param v2: another numpy 2d array
    :return: two numpy 2d array with the same length of the original longer one
    """
    v1_r, v1_c = np.shape(v1)
    v2_r, v2_c = np.shape(v2)
    if v1_r < v2_r:
        v1 = np.append(v1, np.zeros((v2_r-v1_r, v1_c)), axis = 0)
    elif v1_r > v2_r:
        v2 = np.append(v2, np.zeros((v1_r-v2_r, v2_c)), axis = 0)
    else:
        pass
    return v1, v2


def cross_entropy_correct_prob(Y_hat, Y):
    """
    The loss function here is defined as an average over all amino acid
    cross entropy loss (regardless which samples they are from).
    It is good for concatenation of all samples (amino acid string vectors) in a
    batch together.
    What can be done better is https://distill.pub/2017/ctc/
    :param Y_hat: Y_hat is a collection of y_hat, such as y_hat = decoder(
    encoder(x)). y_hat is a softmax output vector for one amino acid.
    :param Y: Y is a collection of Y. y is also an encoded one hot vector for
    one amino acid.
    :return: loss as defined by categorical entropy
    """
    # the zipped list will be truncated by the shorter sequence of Y_hat or Y
    # so the shorter one will be padded with zero vectors by the end
    Y_hat, Y = padding_vector(Y_hat, Y)
    corr_prob = -1.0 * np.sum(np.multiply(Y, np.log(Y_hat)), axis=1)
    return corr_prob

def total_loss(Y_hat, Y):
    return np.sum(cross_entropy_correct_prob(Y_hat, Y)) / len(Y)


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

def tanh_derivative(x):
    """
    https://socratic.org/questions/what-is-the-derivative-of-tanh-x
    :param x: output matrix
    :return: d_tanh
    """
    return 1.0 - np.tanh(x)**2

class RNN(object):
    def __init__(self, input_dim=4, h_dim=100, o_dim=21, bptt_truncate=4, rnn_role=None):
        """
         Initialize the RNN cell including all parameter matrics as W_h, W, and V
        with W_h being for previous hidden status h_t-1
        with W_x being for current input x_t
        with W_o being for output o_t
        The notations followed "Deep Learning" book by Goodfellow, etc.
        :param input_dim: the one-hot encoding of the vocabulary -- it might also be
        an embedded dense encoding dimension
        :param h_dim:
        :param o_dim:
        :param bptt_truncate:
        :param rnn_role: 'decoder', 'encoder' or None
        """
        self.x_dim = input_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.bptt_truncate = bptt_truncate
        self.rnn_role = rnn_role
        self.timesteps = 0
        # param init
        self.W_x = self.xavier_param_init(self.h_dim, self.x_dim, self.x_dim)
        self.W_h = self.xavier_param_init(self.h_dim, self.h_dim, self.h_dim)
        self.W_o = self.xavier_param_init(self.o_dim, self.h_dim, self.h_dim)
        self.b = np.zeros([self.h_dim]).astype(np.float32)
        self.c = np.zeros([self.o_dim]).astype(np.float32)


    def xavier_param_init(self, r_dim, c_dim, prev_layer_dim, use_relu=False):
        """
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        :param r_dim: row dimension
        :param c_dim: column dimension
        :param prev_layer_dim: input vector size (could be output from
        another RNN)
        :param use_relu: False here since used tanh
        :return: init vectors
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
        self.timesteps = len(X)
        # time_step + 1 to add an initial hidden state for the neuron
        H = np.zeros((self.timesteps + 1, self.h_dim))
        O = np.zeros((self.timesteps, self.o_dim))
        for t in range(self.timesteps):
            # update
            H[t] = np.tanh(np.dot(self.W_h, H[t - 1]) + np.dot(self.W_x, X[t]) +
                           self.b)
            # the output itself is based on the current timestep hidden status
            O[t] = softmax(np.dot(self.W_o, H[t]) + self.c)
        return H, O

    def encoding_forward(self, X):
        """
        forward pass as encoding_forward
        :param X: input -- 2d matrix with each row being an one hot encoding
        :return: H, O
        """
        return self.forward(X) # other outputs but the last one can be discarded

    def decoding_forward(self, C, W, eos_vec):
        """
        forward pass as decoding steps
        :param C: the input hidden status or context (thus C)
        :param W: the last output from encoder with the output dimension
        :param eos_vec: use for ending the pass
        :return:
        """
        H = C
        O = W
        while O[self.timesteps] is not eos_vec:
            # update timestep counts
            self.timesteps += 1
            # making spaceholder
            H = np.append(H, np.zeros((1, self.h_dim)), axis = 0)
            O = np.append(O, np.zeros((1, self.o_dim)), axis = 0)
            # update
            # notice that the input X is the last timestep output
            H[self.timesteps] = np.tanh(np.dot(self.W_h, H[self.timesteps - 1]) +
                                        np.dot(self.W_x, O[self.timesteps - 1]) +
                                        self.b)
            # the output itself is based on the current timestep hidden status
            O[self.timesteps] = softmax(np.dot(self.W_o, H[self.timesteps]) +
                                        self.c)
        return H, O

    def backprop_through_time(self, O, Y, k1):
        """
        A truncated back prop through timestep method for passing back loss
        gradient though rnn.

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

        In this implementation, the gradient passing back is separated from
        the gradient descent.
        :param O: normalized output probs
        :param Y: labels one hotcoded
        :param k1: timesteps
        :return: derivatives
        """
        # in case k1 (which would be input sequence length or timesteps) is
        # smaller than predefined self.bptt_truncate
        k1 = self.timesteps
        if k1 < self.bptt_truncate:
            k2 = k1
        else:
            k2 = self.bptt_truncate

        # init placeholder for gradients
        dW_x = np.zeros(self.W_x.shape).astype(np.float32)
        dW_h = np.zeros(self.W_h.shape).astype(np.float32)
        dW_o = np.zeros(self.W_o.shape).astype(np.float32)
        db = np.zeros(self.h_dim).astype(np.float32)
        dc = np.zeros(self.o_dim).astype(np.float32)

        O, Y = padding_vector(O, Y)
        dO = O - Y # only works for one-hot encoding, dO shaped (sample, o_dim)
        for t in reversed(range(k1)):
            # for output weights at timestep t
            # adding time t derivative -- see it as derivative to correct the
            # particular one input at time t
            # By adding up, the gradients will accumulate derivatives at all
            # timesteps, as from all input in the sample sequence
            dW_o += np.outer(dO[t], self.H[t]) # shaped (o_dim, h_dim)
            dc += np.sum(dO[t], axis=0, keepdims=True)
            # for weights for both prev hidden state and input and bias at
            # time t
            # dO[t] shaped 
            dH = np.dot(dO[t], self.W_o[t]) * tanh_derivative(self.H[t])
            # Backprop through time by max(0, t-k2) steps (see above) using
            # derivatives at time t

            for t2 in reversed(range(max(0, t - k2), t + 1)):
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)

    def update(self, dW_x, dW_h, dW_o, db, dc, learning_rate = 0.005):
        """
        update weights by gradient
        :return:
        """
        self.W_x -= learning_rate * dW_x
        self.W_h -= learning_rate * dW_h
        self.W_o -= learning_rate * dW_o
        self.b   -= learning_rate * db
        self.c   -= learning_rate * dc





class Seq2Seq(object):
    def __init__(self):
        # Define two RNN one as encoder and one as decoder
        self.encoder = RNN(input_dim=5, h_dim=100, o_dim=21, bptt_truncate=4,
                           rnn_role='encoder')
        self.decoder = RNN(input_dim=21, h_dim=100, o_dim=21,
                           bptt_truncate=4, rnn_role='decoder')
        self.time_step # this is the k1 by Ilya Sutskever

    def forward(self, X, stop_vec):
        """
        an encoder-decoder forward
        :param X: input sample as an encoding of a sequence
        :return: ec_H, ec_O, dc_H, dc_O
        """
        ec_H, ec_O = self.encoder.encoding_forward(X)
        dc_H, dc_O = self.decoder.decoding_forward(ec_H[-1],
                                                   ec_O[-1],
                                                   stop_vec)
        return ec_H, ec_O, dc_H, dc_O

    def train(self, data, learning_rate=0.005, epoch=100,
              reverse_input=False, stop_vec=encoding(END_OF_SENTENCE, AMINO_ACID)):
        for i in range(0, len(data)):
            X, Y = data[i]
            if reverse_input: X = reverse_seq(X)
            # forward
            ec_H, ec_O, dc_H, dc_O = self.forward(X, stop_vec)
            # evaluate error
            loss = cross_entropy_loss(dc_O, Y)
            # back prop





    def evaluate(self, input):
        pass






