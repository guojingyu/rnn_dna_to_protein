"""
Containing functions to output one-hot encoding for sequence data

Author: Jingyu Guo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from vocabulary import NUCLEOTIDE, AMINO_ACID

# For One-hot encoding
# define two way mapping of chars to integers and an encoding method
def char_to_int(alphabet):
    return dict((c, i) for i, c in enumerate(alphabet))

def int_to_char(alphabet):
    return dict((i, c) for i, c in enumerate(alphabet))

# one hot encode
def encoding(input, alphabet):
    output = np.zeros(len(alphabet))
    output[alphabet.index(input)] = 1
    return output

if __name__ == "__main__":
    # test example
    dna_test_seq = "CTAGT"
    dna_in_int = np.array([char_to_int(NUCLEOTIDE)[n] for n in dna_test_seq])
    dna_encoding = np.array([encoding(n, NUCLEOTIDE) for n in dna_test_seq])
    print(dna_in_int) # [1 0 3 2 0]
    print(dna_encoding)
    # [[ 0.  1.  0.  0.]
    #  [ 1.  0.  0.  0.]
    #  [ 0.  0.  0.  1.]
    #  [ 0.  0.  1.  0.]
    #  [ 1.  0.  0.  0.]]

    ptn_test_seq = "IFGV*"
    ptn_in_int = np.array([char_to_int(AMINO_ACID)[n] for n in ptn_test_seq])
    ptn_encoding = np.array([encoding(n, AMINO_ACID) for n in ptn_test_seq])
    print(ptn_in_int) # [ 8  5  6 18]
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