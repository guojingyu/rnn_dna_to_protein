"""
Containing functions to output one-hot encoding for sequence data

Author: Jingyu Guo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from vocabulary import TRUE_DNA_CODON_TABLE, NUCLEOTIDE, AMINO_ACID, \
    STOP_CODONS, END_OF_SENTENCE, append_eos

# For One-hot encoding
# define two way mapping of chars to integers and an encoding method
def char_to_int(alphabet, add_eos=True, EOS=END_OF_SENTENCE):
    if add_eos:
        alphabet = alphabet + EOS
    return dict((c, i) for i, c in enumerate(alphabet))

def int_to_char(alphabet, add_eos=True, EOS=END_OF_SENTENCE):
    if add_eos:
        alphabet = alphabet + EOS
    return dict((i, c) for i, c in enumerate(alphabet))

# one hot encode
def encoding(input, alphabet, add_eos=True, EOS=END_OF_SENTENCE):
    if add_eos:
        alphabet = alphabet + EOS
    output = np.zeros(len(alphabet))
    output[alphabet.index(input)] = 1
    return output

def decoding(input, alphabet, added_eos=True, EOS=END_OF_SENTENCE):
    pass



# reverse sequence
def reverse_seq(input):
    """
    This is a function to reverse the input sequence
    :param input: input as a numpy sequence object or string
    :return: reversed input
    """
    if type(input) == 'str':
        return input[::-1]
    elif type(input) is np.ndarray:
        return np.flip(input, axis=0)
    else:
        raise Exception("Input not defined for flipping or reversing order "
                        ": " + input)


class BatchGenerator(object):
    def __init__(self, text, batch_size=16, num_unrollings=10,
                 reversed=False):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self.reversed = reversed
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, VOCABULARY_SIZE),
                         dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        if self.reversed:
            batches = batches[::-1]
        return batches



        #
        # if __name__ == "__main__":
        #     train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
        #     valid_batches = BatchGenerator(valid_text, 1, num_unrollings)
        #     train_reversed_batches = BatchGenerator(train_text, batch_size, num_unrollings,
        #                                         True)
        #     valid_reversed_batches = BatchGenerator(valid_text, 1, num_unrollings, True)

# Define a training data generator since we do know the true distribution (the truth codon table) as above
# A typical English sentences from my intuition is about 15 words.
# In this implementation, I will generate sampled data for training the network
def generate_seq_data_by_truth_codon(codon_tbl=TRUE_DNA_CODON_TABLE,
                                     stop_codons=STOP_CODONS,
                                     stop_codon_append_prob=0.05):
    # the DNA length is set to be positive natural number that is product of 3
    # this is enforced since the codon are actually in triplets
    # while it is possible to do it with any length of DNA,
    # just as the language model learned per-char by RNN shown
    # in Andrej Karpathy's blog post.
    # uniform sampling in [ 3,  6,  9, 12, 15, 18, 21, 24, 27, 30]
    DNA_length = np.random.choice(np.arange(3, 30 + 1, 3))
    keys = list(codon_tbl.keys())
    num_codon = len(keys)

    j = 0
    dna_seq = ""
    ptn_seq = ""
    while j < DNA_length / 3:
        codon = keys[np.random.choice(num_codon, 1)[0]]
        if codon not in stop_codons:
            dna_seq += codon
            ptn_seq += TRUE_DNA_CODON_TABLE[codon]
            j += 1
    if stop_codon_append_prob > np.random.uniform():
        stop_c = stop_codons[np.random.choice(np.arange(len(stop_codons)), 1)[0]]
        dna_seq += stop_c
        ptn_seq += TRUE_DNA_CODON_TABLE[stop_c]

    return [dna_seq, ptn_seq]

def generate_data(n=1000, encoded=True, codon_tbl=TRUE_DNA_CODON_TABLE,
                           stop_codons=STOP_CODONS,
                           stop_codon_append_prob=0.05):
    data = []
    for i in range(n):
        dna_seq, ptn_seq = generate_seq_data_by_truth_codon(codon_tbl,
                                                            stop_codons,
                                                            stop_codon_append_prob)
        if encoded:
            data.append([np.array([encoding(n, NUCLEOTIDE) for n in
                                   dna_seq]),
                         np.array([encoding(n, AMINO_ACID) for n in
                                   ptn_seq])])
        else:
            data.append([dna_seq, ptn_seq])
    return data

if __name__ == "__main__":
    # test example
    dna_test_seq = append_eos("CTAGT")
    dna_in_int = np.array([char_to_int(NUCLEOTIDE)[n] for n in
                           dna_test_seq])
    dna_encoding = np.array([encoding(n, NUCLEOTIDE) for n in
                             dna_test_seq])
    print(dna_in_int) # [1 3 0 2 3 4]
    print(dna_encoding)
    # [[ 0.  1.  0.  0.  0.]
    # [ 0.  0.  0.  1.  0.]
    # [ 1.  0.  0.  0.  0.]
    # [ 0.  0.  1.  0.  0.]
    # [ 0.  0.  0.  1.  0.]
    # [ 0.  0.  0.  0.  1.]]

    ptn_test_seq = "IFGV*"
    ptn_in_int = np.array([char_to_int(AMINO_ACID, False)[n] for n in
                           ptn_test_seq])
    ptn_encoding = np.array([encoding(n, AMINO_ACID, False) for n in
                             ptn_test_seq])
    print(ptn_in_int) # [ 8  5  6 18  0]
    print(ptn_encoding)
    # [[ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.
    #  0. 0.  0.  0.]
    # [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    # 0. 0.  0.  0.]
    # [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    # 0. 0.  0.  0.]
    # [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    # 0. 1.  0.  0.]
    # [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    # 0. 0.  0.  0.]]


    # Using keras is much easier to get a zero-starting numeric representation of
    # categorical list into one-hot coding, such as:
    # from keras.utils import to_categorical
    # encoded = to_categorical([1,3,0,1,4,2])
    # array([[ 0.,  1.,  0.,  0.,  0.],
    #        [ 0.,  0.,  0.,  1.,  0.],
    #        [ 1.,  0.,  0.,  0.,  0.],
    #        [ 0.,  1.,  0.,  0.,  0.],
    #        [ 0.,  0.,  0.,  0.,  1.],
    #        [ 0.,  0.,  1.,  0.,  0.]])

    # scikit-learn has a similar one-hot encoding tool as
    # sklearn.preprocessing.OneHotEncoder


    print(generate_seq_data_by_truth_codon(codon_tbl=TRUE_DNA_CODON_TABLE,
                                     stop_codons=STOP_CODONS,
                                     stop_codon_append_prob=0.05))
    print(generate_data(n=10, encoded=False))