from framework.layers.absLoss import ABSLoss
from framework.layers.crossentropyloss import CrossEntropyLoss
from framework.layers.embedding import Embedding
from framework.layers.linear import Linear
from framework.layers.mseLoss import MSELoss
from framework.layers.relu import ReLU
from framework.layers.rnncell import RNNCell
from framework.layers.sequential import Sequential
from framework.layers.sigmoid import Sigmoid
from framework.optimization.sgd import SGD


class LinearClassificationOfExpressions():
    #Модель нейронной сети для классификации обзоров c RNN
    def __init__(self,vocab,len_tokens):
        #self.embed = Embedding(vocab_size=len(vocab), dim=30)
        self.model = Sequential([RNNCell(n_inputs=len_tokens, n_hidden=400, n_output=len(vocab)), Linear(30,1), Sigmoid()])
        self.criterion = ABSLoss()
        self.optim = SGD(parameters= self.model.get_parameters(), alpha=0.05)