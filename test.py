from utilities import Corpus, LDA
import numpy as np


data_path = '../dataset/ap.txt'
vocab_path = '../dataset/vocab.txt'

with open(data_path, 'r') as input_file:
    corpus = input_file.read()

with open(vocab_path, 'r') as vocab_file:
    vocabs = vocab_file.readlines()

for i in range(len(vocabs)):
    vocabs[i] = vocabs[i][:-1]

corpus_instance = Corpus(corpus, 10)
lda_instance = LDA(corpus_instance, 1, 0.1)
lda_instance.train(max_iter=30)

theta = lda_instance.get_theta()
phi = lda_instance.get_phi()
perplexity = lda_instance.perplexity()
print(theta)
print(phi)
print(perplexity)

np.savetxt('../results/theta.txt', theta)
np.savetxt('../results/phi.txt', phi)
np.savetxt('../results/perplexity.txt', perplexity)

