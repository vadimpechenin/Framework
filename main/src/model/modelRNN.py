from lightweightDeepLearningFramework.layers.crossentropyloss import CrossEntropyLoss
from lightweightDeepLearningFramework.layers.embedding import Embedding
from lightweightDeepLearningFramework.layers.rnncell import RNNCell
from lightweightDeepLearningFramework.optimization.sgd import SGD


class ModelRNN():
    #Модель нейронной сети с RNN
    def __init__(self,vocab):
        self.embed = Embedding(vocab_size=len(vocab),dim=512)
        self.model = RNNCell(n_inputs=512, n_hidden=512, n_output=len(vocab))
        self.criterion = CrossEntropyLoss()
        self.optim = SGD(parameters= self.model.get_parameters() + self.embed.get_parameters(), alpha=0.05)