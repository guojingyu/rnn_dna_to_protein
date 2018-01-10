### FROM DNA TO PROTEIN

#### Note
The current version is not learning well on this 'looking-simple' task. The
cross entropy loss is reducing over epoch but is not going down too much. The
training of this encoding-decoding is still suffering from gradient vanish or
explosion.

I expect approaching 0 training loss in ideal situation due to unlimited no
noise data available. It is possible that simple naive RNN based encoder
decoder will not work well for this issue. LSTM may be worthy trying. For now,
it still needs more investigation and work.

#### Overview
This is a RNN (sequence to sequence or encoder-decoder) demo implementation to
learn general rules of central dogma of molecular biology. More specifically, it is trying to learn the [DNA codons](https://en.wikipedia.org/wiki/DNA_codon_table)
that is used to translate the DNA sequence to protein sequence (like a
dictionary). The data is generated from a truth codon table (unlimited) but it
can also be from public sequence repos as below.

#### Sample Data
CFTR:
mRNA (cDNA): [NCBI fasta link](https://www.ncbi.nlm.nih.gov/nuccore/NM_000492.3?report=fasta&log$=seqview&format=text&sat=4&satkey=207527635&from=133&to=4575)

Protein: [NCBI fasta link](https://www.ncbi.nlm.nih.gov/protein/NP_000483.3?report=fasta&log$=seqview&format=text&sat=4&satkey=207527635)

#### References
1. [Sequence to sequence learning](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
2. [Backprop through time](https://machinelearningmastery.com/gentle-introduction-backpropagation-time/)
3. [Xavier weight initialization](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
4. The loss function is defined as an average over all amino acid cross entropy
 loss (regardless which samples they are from). For sequence to sequence
 learning, the potential different length between the predicted sequence and
 the 'label' sequence can be a challenging issue. For now, I just simply pad
 the shorter sequence with zeros (with the expectation that all zeros will lead
  to loss with any one-hot encoding and it is the best for the learner to not
  output longer sequence -- but it may not work well so far)
     [What can be done better/differently](https://distill.pub/2017/ctc/)
5. [clipping gradient by its norm](https://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf)
6. [softmax implementation note](http://cs231n.github.io/linear-classify/#softmax)

