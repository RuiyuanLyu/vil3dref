import numpy as np
def get_glove_word2vec(file='datasets/pretrained/wordvecs/glove.42B.300d.txt'):
    word2vec = {}
    with open(file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            word2vec[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    return word2vec
