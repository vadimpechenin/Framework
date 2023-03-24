#Класс слоя, выход функция от входов (mse)
from framework.layers.layer import Layer


class ABSLoss(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return (pred - target).sum(0)