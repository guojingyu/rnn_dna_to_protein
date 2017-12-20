"""
Adding known knowledge about codons for final check

Author: Jingyu Guo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Please notice that the 'TAA', 'TAG', and 'TGA' are so called stop codons,
# they do not encode any amino acid but indicate the STOP which is a '*' here.
# 'ATG' on the other hand encodes Methionine or Met or 'M', which always is the
# start codon.
# Also please notice more than one codon could encode the same amino acid,
# such as 'GCA', 'GCC', 'GCG', and 'GCT' are all Alanine or 'A'.
TRUE_DNA_CODON_TABLE = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
    'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
}


# Now to build the transcription table part
def transcribe(seq):
    """
    The general transcription rule is simple:
    DNA  RNA
     A -> A
     T -> U (uracil)
     C -> C
     G -> G
    """
    rna_seq = seq.replace('T', 'U')
    return rna_seq


TRANSCRIPTION_TABLE = {"A": transcribe("A"), "T": transcribe("T"),
                       "C": transcribe("C"), "G": transcribe("G")}
# print(TRANSCRIPTION_TABLE)
#  {'A': 'A', 'T': 'U', 'C': 'C', 'G': 'G'}

# Include nucleotide alphabet
NUCLEOTIDE = "".join(set(TRANSCRIPTION_TABLE.keys()))

# Also included are some amino acid alphabet
AMINO_ACID = "".join(sorted(list(set(TRUE_DNA_CODON_TABLE.values()))))
# print(NUCLEOTIDE, AMINO_ACID)
# *ACDEFGHIKLMNPQRSTVWY