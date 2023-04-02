import cupy as cp

def openFileArticlex(fileName):
    f = open(fileName)
    fileСontents = f.readlines()
    f.close()
    return fileСontents


def load_dataIMDB_NP_CP(fileName, fileNameLabels,rawLength=1000, sentenseLength=100):
    #Основной метод предварительно подготовки слов
    #Метод для загрузки отзывов и рейтингов, создания векторного представления слов

    #rawReviews = self.openFile(self.reviewsName)
    #rawLabels  = self.openFile(self.labelsName)

    rawReviews = openFileArticlex(fileName)
    rawLabels  = openFileArticlex(fileNameLabels)

    tokens = list()
    for line in rawReviews[0:rawLength]:
        result = line.lower().replace("\n", "").split(" ")
        if len(result)>sentenseLength:
            result = result[0:sentenseLength]
        tokens.append(result)

    new_tokens = list()
    for line in tokens:
        new_tokens.append(['-'] * (sentenseLength - len(line)) + line)

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
    data_cp = cp.array(indices)
    targetDataset = list()
    for label in rawLabels[0:rawLength]:
        if label == 'positive\n':
            targetDataset.append(1)
        else:
            targetDataset.append(0)

    return vocab, data, data_cp, targetDataset