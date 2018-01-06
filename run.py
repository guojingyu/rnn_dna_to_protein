"""
Running encoder decoder seq-2-seq learning codon

Author Jingyu Guo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from data_processing import encoding, decoding, reverse_seq, generate_data, \
    get_sequences_from_fasta_file, generate_seq_data_by_sampling_seq
from vocabulary import TRUE_DNA_CODON_TABLE, NUCLEOTIDE, AMINO_ACID, \
    STOP_CODONS, END_OF_SENTENCE
from rnn import RNN, Seq2Seq
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

def train_seq_2_seq(encoder, decoder, stop_vec, data, learning_rate, epoch,
                    print_interval):
    """

    :return:
    """
    learner = Seq2Seq(encoder, decoder)
    learner.train(data, learning_rate=learning_rate, epoch=epoch,
              reverse_input=False,
              stop_vec=stop_vec,
              print_interval=print_interval)
    return learner


def get_alignment(pred, label, matching=2, mismatching=-1, gap=0.5,
                  extend=0.1, print_out=True):
    """
    Biopython Tutorial and Cookbook:
    http://biopython.org/DIST/docs/tutorial/Tutorial.html
    global alignments with the maximum similarity score
    Matching characters are given 2 points
    mismatching character -1
    open gap 0.5
    extend sequence 0.1
    :param pred: predicted seq
    :param label: testing label seq
    :param matching: matching score
    :param mismatching: mismatching score
    :param gap: gap score
    :param extend: extend score
    :param print_out: boolean option to printout alignment
    :return: alignment object
    """
    alignments = pairwise2.align.globalms(pred, label, 2, -1, -0.5, -0.1)
    if print_out:
        for a in alignments:
            print(format_alignment(*a))
    return alignments


if __name__ == "__main__":
    # setting param
    encoded = True
    training_size = 1000
    learning_rate = 0.005
    epoch = 100
    print_interval = 100
    dna_stop_vec = encoding(END_OF_SENTENCE, NUCLEOTIDE)
    ptn_stop_vec = encoding(END_OF_SENTENCE, AMINO_ACID)

    # training data prep
    data = generate_data(n=training_size, encoded=encoded)

    # construct learner and train
    encoder = RNN(input_dim=5, h_dim=100, o_dim=21, bptt_truncate=4,
                  rnn_role='encoder')
    decoder = RNN(input_dim=21, h_dim=100, o_dim=21, bptt_truncate=4,
                  rnn_role='decoder')
    seq_2_seq_learner = train_seq_2_seq(encoder, decoder, dna_stop_vec, data,
                                        learning_rate, epoch, print_interval)

    # prepare test data
    test_cdna_file = "data/cDNA_CFTR.fa"
    test_cdna_seq_dict = get_sequences_from_fasta_file(test_cdna_file)
    dna_sequences = [seq_obj.seq for seq_obj in test_cdna_seq_dict.values()]
    print(dna_sequences[0])
    dna_length = len(dna_sequences[0])
    test_aa_file = "data/amino_acid_CFTR.fa"
    test_aa_seq_dict = get_sequences_from_fasta_file(test_aa_file)
    aa_sequences = [seq_obj.seq for seq_obj in test_aa_seq_dict.values()]
    print(aa_sequences[0])
    aa_length = len(dna_sequences[0])
    test_data = generate_seq_data_by_sampling_seq(dna_sequences[0],
                                              aa_sequences[0])
    for i in range(0, len(test_data)):
        X, Y = test_data[i]
        ec_pred = seq_2_seq_learner.evaluate(X, dna_stop_vec)
        label = "".join([decoding(v, AMINO_ACID) for v in Y])
        pred = "".join([decoding(v, AMINO_ACID) for v in ec_pred])
        if i % 50 == 0:
            get_alignment(pred, label)
        else:
            get_alignment(pred, label, False)

