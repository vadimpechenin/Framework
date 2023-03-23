from framework.layers.crossentropyloss import CrossEntropyLoss
from framework.layers.embedding import Embedding
from framework.layers.linear import Linear
from framework.layers.relu import ReLU
from framework.layers.sequential import Sequential
from framework.layers.sigmoid import Sigmoid
from framework.optimization.sgd import SGD


class LinearClassificationOfExpressions():
    #Модель простой нейронной сети для классификации обзоров
    def __init__(self,vocab):
        self.model = Sequential([Embedding(vocab_size=len(vocab), dim=1), ReLU(), Linear(1,1), Sigmoid()])
        self.criterion = CrossEntropyLoss()
        self.optim = SGD(parameters= self.model.get_parameters(), alpha=0.05)