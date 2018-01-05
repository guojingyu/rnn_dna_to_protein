"""
Running encoder decoder seq-2-seq learning codon

Author Jingyu Guo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from data_processing import encoding, decoding, reverse_seq
from vocabulary import TRUE_DNA_CODON_TABLE, NUCLEOTIDE, AMINO_ACID, \
    STOP_CODONS, END_OF_SENTENCE
from rnn import RNN, Seq2Seq

def generate_data():
    """

    :return:
    """
    pass

def run():
    """

    :return:
    """
    #TODO
    encoder = RNN(input_dim=5, h_dim=100, o_dim=21, bptt_truncate=4,
                           rnn_role='encoder')
    decoder = RNN(input_dim=21, h_dim=100, o_dim=21,
                           bptt_truncate=4, rnn_role='decoder')
    learner = Seq2Seq(encoder, decoder)
    pass

if __name__ == "__main__":
    run()
