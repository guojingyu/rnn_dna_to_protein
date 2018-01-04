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
            H[t] = np.tanh(np.dot(self.W_h, H[t - 1]) +
                           np.dot(self.W_x, X[t]) +
                           self.b) # H[t] shaped (self.h_dim, 1)
            # the output itself is based on the current timestep hidden status
            O[t] = softmax(np.dot(self.W_o, H[t]) + self.c) # O[t] (self.o_dim, 1)
        return H, O

    def encoding_forward(self, X):
        """
        forward pass as encoding_forward
        :param X: input -- 2d matrix with each row being an one hot encoding
        :return: H
        """
        H, _ = self.forward(X)
        return H

    def decoding_forward(self, C, eos_vec):
        """
        forward pass as decoding steps
        :param C: the input hidden status or context (thus C)
        :param eos_vec: use for ending the pass
        :return:
        """
        H = np.array([C])
        O = np.zeros((1, self.o_dim))
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

    def backprop_through_time(self, X, Y, H, O):
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
        :param X: input
        :param Y: labels one hotcoded
        :param O: normalized output probs
        :param H: hidden states after forward pass with X, Y
        :return: gradients to update weights
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

        # gradient for the softmax
        O, Y = padding_vector(O, Y)
        dO = O - Y # only works for one-hot encoding, dO shaped (sample, o_dim)
        for t in reversed(range(k1)):
            # for output weights at timestep t
            # adding time t derivative -- see it as derivative to correct the
            # particular one input at time t
            # By adding up, the gradients will accumulate derivatives at all
            # timesteps, as from all input in the sample sequence
            dW_o += np.outer(dO[t], H[t]) # shaped (o_dim, h_dim)
            dc += np.sum(dO[t], axis=0, keepdims=True)
            # for gradients for both prev hidden state and input and bias at
            # time t
            # dO[t] shaped (1, o_dim) W_o shape (o_dim, h_dim)
            dH_t = np.dot(dO[t], self.W_o) * tanh_derivative(H[t]) #dH_t (1, h_dim)
            # Backprop through time from time t by max(1, t-k2+1) steps (see
            # above) using hidden derivatives at time t
            # this is a tricky bit: there is one time step added for hidden
            # states in forward pass, thus the max is not thresheld by 0 but
            # by 1, thus the real first input is at timestep 1.
            for t2 in reversed(range(max(1, t - k2 + 1), t)):
                dW_h += np.outer(dH_t, H[t2-1]) # -1 for prev step
                # shape (h_dim, h_dim)
                dW_x += np.outer(dH_t, X[t2]) # shape (h_dim, x_dim)
                db += np.sum(dH_t[t2], axis=0, keepdims=True)
                # Update hidden gradients for one previous timestep in back prop
                dH_t = np.dot(dH_t, self.W_h) * tanh_derivative(H[t2-1])
        return dW_x, dW_h, dW_o, db, dc


    def encoding_backprop_through_time(self, X, dContext, H):
        """
        Unlike vanilla RNN, encoder does not have Y to supervise the
        learning. Thus the backprop-tt will needed to be revised to only
        consider 1 supervising source (dContext) as the gradient of the context
        export.
        :param X: input
        :param dContext: dContext will be the dW_h from decoder bptt algorithm
        :param H: hidden states after forward pass with X
        :return: gradients to update weights
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
        db = np.zeros(self.h_dim).astype(np.float32)

        dH_t = dContext
        for t in reversed(range(k1)):
            # Backprop through time from time t by max(1, t-k2+1) steps (see
            # above) using hidden derivatives at time t
            # this is a tricky bit: there is one time step added for hidden
            # states in forward pass, thus the max is not thresheld by 0 but
            # by 1, thus the real first input is at timestep 1.
            for t2 in reversed(range(max(1, t - k2 + 1), t)):
                dW_h += np.outer(dH_t, H[t2 - 1])  # -1 for prev step
                # shape (h_dim, h_dim)
                dW_x += np.outer(dH_t, X[t2])  # shape (h_dim, x_dim)
                db += np.sum(dH_t[t2], axis=0, keepdims=True)
                # Update hidden gradients for one previous timestep in back prop
                dH_t = np.dot(dH_t, self.W_h) * tanh_derivative(H[t2 - 1])
        return dW_x, dW_h, db


    def decoding_backprop_through_time(self, X, Y, H, O):
        """
        Like vanilla RNN, decoder has O as the input X. Thus the
        backprop-tt can stay the same
        :param X: input
        :param Y: labels one hotcoded
        :param O: normalized output probs
        :param H: hidden states after forward pass with X, Y
        :return: gradients to update weights
        """
        return self.backprop_through_time(X, Y, H, O)

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

    def train(self, data, learning_rate=0.005, epoch=100, print_interval=100):
        """
        train the vanilla rnn method
        :param X: Input
        :param Y: Input lable
        :return: loss records
        """
        loss = []
        for n in range(epoch):
            for i in range(0, len(data)):
                X, Y = data[i]
                if reverse_input: X = reverse_seq(X)
                # forward
                H, O = self.forward(X)
                # back prop
                dW_x, dW_h, dW_o, db, dc = self.backprop_through_time(X, Y, H, O)
                # update
                self.update(dW_x, dW_h, dW_o, db, dc, learning_rate)
            # evaluate error
            current_loss = total_loss(O, Y)
            loss.append((n, current_loss))
            if n % print_interval == 0:
                print("Current Loss: %s".format())
        return loss


    def evaluate(self, X):
        """
        evaluate function
        :param X: input -- 2d matrix with each row being an one hot encoding of X length
        :return: output -- 2d matrix with each row being an one hot encoding of O length
        """
        _, O = self.forward(X)
        pred = np.zeros(np.shape(O))
        one_index = np.argmax(O, axis=1)
        pred[np.arange(len(pred)), one_index] = 1.0
        return pred


class Seq2Seq(object):
    def __init__(self, encoder, decoder):
        # Define two RNN one as encoder and one as decoder
        self.encoder = encoder
        self.decoder = decoder
        self.timesteps = 0

    def forward(self, X, stop_vec):
        """
        an encoder-decoder forward
        :param X: input sample as an encoding of a sequence
        :return: ec_H, ec_O, dc_H, dc_O
        """
        ec_H = self.encoder.encoding_forward(X)
        dc_H, dc_O = self.decoder.decoding_forward(ec_H[-1], stop_vec)
        self.timesteps = self.encoder.timesteps + self.encoder.timesteps
        return ec_H, dc_H, dc_O

    def train(self, data, learning_rate=0.005, epoch=100,
              reverse_input=False,
              stop_vec=encoding(END_OF_SENTENCE, AMINO_ACID),
              print_interval=100):
        loss = []
        for n in range(epoch):
            for i in range(0, len(data)):
                X, Y = data[i]
                if reverse_input: X = reverse_seq(X)
                # forward
                ec_H, dc_H, dc_O = self.forward(X, stop_vec)
                # back prop
                dW_x, dW_h, dW_o, db, dc = self.backprop_through_time(X, Y, H, O)
                # update
                self.update(dW_x, dW_h, dW_o, db, dc, learning_rate)
            # evaluate error
            current_loss = total_loss(O, Y)
            loss.append((n, current_loss))
            if n % print_interval == 0:
                print("Current Loss: %s".format())
        return loss


    def evaluate(self, X, stop_vec):
        """
        evaluate function
        :param X: input -- 2d matrix with each row being an one hot encoding of X length
        :return: output -- 2d matrix with each row being an one hot encoding of O length
        """
        _, _, dc_O = self.forward(X, stop_vec)
        pred = np.zeros(np.shape(dc_O))
        one_index = np.argmax(dc_O, axis=1)
        pred[np.arange(len(pred)), one_index] = 1.0
        return pred




RNN(input_dim=5, h_dim=100, o_dim=21, bptt_truncate=4,
                           rnn_role='encoder')
RNN(input_dim=21, h_dim=100, o_dim=21,
                           bptt_truncate=4, rnn_role='decoder')


