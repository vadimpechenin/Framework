from framework.layers.crossentropyloss import CrossEntropyLoss
from framework.layers.dropout import Dropout
from framework.layers.embedding import Embedding
from framework.layers.rnncell import RNNCell
from framework.layers.sequential import Sequential
from framework.optimization.sgd import SGD


class ModelRNN():
    #Модель нейронной сети с RNN c дропаутом
    def __init__(self,vocab):
        self.embed = Embedding(vocab_size=len(vocab),dim=400)
        self.model = RNNCell(n_inputs=400, n_hidden=300, n_output=2)
        self.criterion = CrossEntropyLoss()
        self.optim = SGD(parameters= self.model.get_parameters() + self.embed.get_parameters(), alpha=0.05)