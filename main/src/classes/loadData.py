import numpy as np


def load_data(fileName):
    #Подготовка входных данных
    f = open(fileName, 'r')
    raw = f.readlines()
    f.close()

    tokens = list()
    for line in raw[0:1000]:
        tokens.append(line.lower().replace("\n", "").split(" ")[1:])

    new_tokens = list()
    for line in tokens:
        new_tokens.append(['-'] * (6 - len(line)) + line)

    tokens = new_tokens

    vocab = set()
    for sent in tokens:
        for word in sent:
            vocab.add(word)

    vocab = list(vocab)

    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i


    def words2indices(sentence):
        idx = list()
        for word in sentence:
            idx.append(word2index[word])
        return idx


    indices = list()
    for line in tokens:
        idx = list()
        for w in line:
            idx.append(word2index[w])
        indices.append(idx)

    data = np.array(indices)

    return data, vocab