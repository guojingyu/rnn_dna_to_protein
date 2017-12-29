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



def cross_entropy_loss(self, Y_hat, Y):
    """
    The loss function here is defined as an average over all amino acid
    cross entropy loss (regardless which samples they are from).
    It is good for concatenation of all samples (amino acid string vectors) in a
    batch together.
    :param Y_hat: Y_hat is a collection of y_hat, such as y_hat = decoder(
    encoder(x)). y_hat is a softmax output vector for one amino acid.
    :param Y: Y is a collection of Y. y is also an encoded one hot vector for
    one amino acid.
    :return: loss as defined by categorical entropy
    """
    loss = 0.0
    # the zipped list will be truncated by the shorter sequence of Y_hat or Y
    for y_hat, y in zip(Y_hat, Y):
        loss += -1.0 * np.sum(np.multiply(y, np.log(y_hat)))
    return loss/len(Y)

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
    :param x: loss or
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
        self.time_step = 0
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
        self.time_step = len(X)
        # time_step + 1 to add an initial hidden state for the neuron
        H = np.zeros((self.time_step + 1, self.h_dim))
        O = np.zeros((self.time_step, self.o_dim))
        for t in range(self.time_step):
            # update
            H[t] = np.tanh(self.W_h.dot(H[t - 1]) + self.W_x.dot(X[t]))
            # the output itself is based on the current timestep hidden status
            O[t] = softmax(self.W_o.dot(H[t]))
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
        while O[self.time_step] is not eos_vec:
            # update timestep counts
            self.time_step += 1
            # making spaceholder
            H = np.append(H, np.zeros((1, self.h_dim)), axis = 0)
            O = np.append(O, np.zeros((1, self.o_dim)), axis = 0)
            # update
            # notice that the input X is the last timestep output
            H[self.time_step] = np.tanh(self.W_h.dot(H[self.time_step - 1]) +
                                        self.W_x.dot(O[self.time_step - 1]))
            # the output itself is based on the current timestep hidden status
            O[self.time_step] = softmax(self.W_o.dot(H[self.time_step]))
        return H, O

    def backprop_through_time(self, error, k1):
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
        """
        # implement configuration of TBPTT(k1,k2), where k1<k2<timesteps:
        # Multiple updates are performed per sequence which can accelerate training.
        # in case k1 (which would be input sequence length or timesteps) is
        # smaller than predefined self.bptt_truncate
        if k1 < self.bptt_truncate:
            k2 = k1
        else:
            k2 = self.bptt_truncate

        # init placeholder for gradients
        dL_dW_x = np.zeros(self.W_x.shape).astype(np.float32)
        dL_dW_h = np.zeros(self.W_h.shape).astype(np.float32)
        dL_dW_o = np.zeros(self.W_o.shape).astype(np.float32)
        dL_db = np.zeros([self.h_dim]).astype(np.float32)
        dL_dc = np.zeros([self.o_dim]).astype(np.float32)

        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[
                             ::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)

        for i in range(k1) # k1
        for i in range(bptt_truncate): # k2


    def update(self, dL_dW_x, dL_dW_h, dL_dW_o, learning_rate = 0.005):
        """
        update weights by gradient
        :param dL_dW_x:
        :param dL_dW_h:
        :param dL_dW_o:
        :param learning_rate:
        :return:
        """
        self.W_x -= learning_rate * dL_dW_x
        self.W_h -= learning_rate * dL_dW_h
        self.W_o -= learning_rate * dL_dW_o





class Seq2Seq(object):
    def __init__(self):
        # Define two RNN one as encoder and one as decoder
        self.encoder = RNN(input_dim=5, h_dim=100, o_dim=21, bptt_truncate=4,
                           rnn_role='encoder')
        self.decoder = RNN(input_dim=21, h_dim=100, o_dim=21,
                           bptt_truncate=4, rnn_role='decoder')
        self.time_step # this is the k1 by Ilya Sutskever


    def train(self, data, learning_rate=0.005, epoch=100,
              reverse_input=False, stop_vec=encoding(END_OF_SENTENCE, AMINO_ACID)):
        for i in range(0, len(data)):
            X, Y = data[i]
            if reverse_input: X = reverse_seq(X)
            # forward
            ec_H, ec_O = self.encoder.encoding_forward(X)
            dc_H, dc_O = self.decoder.decoding_forward(ec_H[-1],
                                                       ec_O[-1],
                                                       stop_vec)
            # evaluate error
            loss = cross_entropy_loss(dc_O, Y)
            # back prop error




    def evaluate(self, input):
        pass






