from framework.layers.absLoss import ABSLoss
from framework.layers.embedding import Embedding
from framework.layers.linear import Linear
from framework.layers.lstmcell import LSTMCell
from framework.layers.sequential import Sequential
from framework.layers.sigmoid import Sigmoid
from framework.optimization.sgd import SGD


class LSTMWithEmbeddingClassificationOfExpressions():
    #Модель нейронной сети для классификации обзоров c LSTM
    def __init__(self,vocab,len_tokens):
        self.embed = Embedding(vocab_size=len(vocab), dim=len_tokens)
        self.LSTM = LSTMCell(n_inputs=len_tokens, n_hidden=50, n_output=len(vocab)),
        self.model = Sequential([Linear(len(vocab),1), Sigmoid()])
        self.criterion = ABSLoss()
        self.optim = SGD(parameters= self.model.get_parameters()+self.LSTM[0].get_parameters()+ self.embed.get_parameters(), alpha=0.05)