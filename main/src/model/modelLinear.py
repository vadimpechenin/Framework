from framework.layers.embedding import Embedding
from framework.layers.linear import Linear
from framework.layers.mseLoss import MSELoss
from framework.layers.sequential import Sequential
from framework.layers.sigmoid import Sigmoid
from framework.layers.tanh import Tanh
from framework.optimization.sgd import SGD


class LinearClassification():
    #Модель простой нейронной сети для классификации обзоров
    def __init__(self, pl):
        if (pl==0):
            self.model = Sequential([Linear(2, 3), Tanh(), Linear(3, 1), Sigmoid()])
        else:
            self.model = Sequential([Embedding(vocab_size=5, dim=3), Tanh(), Linear(3,1), Sigmoid()])
        self.criterion = MSELoss()
        self.optim = SGD(parameters= self.model.get_parameters(), alpha=0.5)