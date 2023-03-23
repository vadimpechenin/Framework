from framework.layers.crossentropyloss import CrossEntropyLoss
from framework.layers.embedding import Embedding
from framework.layers.linear import Linear
from framework.layers.mseLoss import MSELoss
from framework.layers.relu import ReLU
from framework.layers.sequential import Sequential
from framework.layers.sigmoid import Sigmoid
from framework.optimization.sgd import SGD


class LinearClassificationOfExpressions():
    #Модель простой нейронной сети для классификации обзоров
    def __init__(self,vocab,len_tokens):
        self.embed = Embedding(vocab_size=len(vocab), dim=30)
        self.model = Sequential([Linear(len_tokens,30), ReLU(), Linear(30,1), Sigmoid()])
        self.criterion = MSELoss()
        self.optim = SGD(parameters= self.model.get_parameters(), alpha=0.05)