import unittest

import numpy as np

from classes.timeIt import timeit
from framework.tensor.tensor import Tensor
from framework.layers.crossentropyloss import CrossEntropyLoss
from framework.layers.embedding import Embedding
from framework.layers.linear import Linear
from framework.layers.mseLoss import MSELoss
from framework.layers.sequential import Sequential
from framework.layers.sigmoid import Sigmoid
from framework.layers.tanh import Tanh
from framework.optimization.sgd import SGD
import time

class LayersTest(unittest.TestCase):
    # Базовые тесты слоев фрейморка
    @timeit("Test 1")
    def test1(self):
        # 1 тесты по построению нейронной сети с полученным классом тензор
        # Как было раньше

        np.random.seed(0)

        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        target = np.array([[0], [1], [0], [1]])

        weights_0_1 = np.random.rand(2, 3)
        weights_1_2 = np.random.rand(3, 1)

        for i in range(10):
            # Predict
            layer_1 = data.dot(weights_0_1)
            layer_2 = layer_1.dot(weights_1_2)

            # Compare
            diff = (layer_2 - target)
            sqdiff = (diff * diff)
            loss = sqdiff.sum(0)  # mean squared error loss

            # Learn: this is the backpropagation piece
            layer_1_grad = diff.dot(weights_1_2.transpose())
            weight_1_2_update = layer_1.transpose().dot(diff)
            weight_0_1_update = data.transpose().dot(layer_1_grad)

            weights_1_2 -= weight_1_2_update * 0.1
            weights_0_1 -= weight_0_1_update * 0.1
            print(loss[0])

    @timeit("Test 2")
    def test2(self):
        # 2 Как теперь с Tensor
        np.random.seed(0)

        data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
        target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

        w = list()
        w.append(Tensor(np.random.rand(2, 3), autograd=True))
        w.append(Tensor(np.random.rand(3, 1), autograd=True))

        for i in range(10):

            # Predict
            pred = data.mm(w[0]).mm(w[1])

            # Compare
            loss = ((pred - target) * (pred - target)).sum(0)

            # Learn
            loss.backward(Tensor(np.ones_like(loss.data)))

            for w_ in w:
                w_.data -= w_.grad.data * 0.1
                w_.grad.data *= 0

            print(loss)

    @timeit("Test 3")
    def test3(self):
        # 3 Как теперь с оптимизатором градиентного спуска

        np.random.seed(0)

        data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
        target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

        w = list()
        w.append(Tensor(np.random.rand(2, 3), autograd=True))
        w.append(Tensor(np.random.rand(3, 1), autograd=True))

        optim = SGD(parameters=w, alpha=0.1)

        for i in range(10):
            # Predict
            pred = data.mm(w[0]).mm(w[1])

            # Compare
            loss = ((pred - target) * (pred - target)).sum(0)

            # Learn
            loss.backward(Tensor(np.ones_like(loss.data)))

            optim.step()

            print(loss)

    @timeit("Test 4")
    def test4(self):
        # 4 Как теперь с оптимизатором градиентного спуска и слоями
        np.random.seed(0)
        print('С нелинейными слоями')

        data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
        target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

        # Архитектура сети
        model = Sequential([Linear(2, 3), Tanh(), Linear(3, 1), Sigmoid()])
        criterion = MSELoss()

        optim = SGD(parameters=model.get_parameters(), alpha=1)

        for i in range(10):
            # Predict
            pred = model.forward(data)

            # Compare
            loss = criterion.forward(pred, target)

            # Learn
            loss.backward(Tensor(np.ones_like(loss.data)))

            optim.step()

            print(loss)

    @timeit("Test 5")
    def test5(self):
        # 6. Проверка индексирования
        x1 = Tensor(np.eye(5), autograd=True)
        x1.index_select(Tensor([[1, 2, 3], [2, 3, 4]])).backward()
        print(x1.grad)

    @timeit("Test 6")
    def test6(self):
        # 7 Реализация со слоем индексирования
        np.random.seed(0)
        data = Tensor(np.array([1, 2, 1, 2]), autograd=True)
        target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

        embed = Embedding(5, 3)
        model = Sequential([embed, Tanh(), Linear(3, 1), Sigmoid()])
        criterion = MSELoss()

        optim = SGD(parameters=model.get_parameters(), alpha=0.5)
        print('Embedding')
        for i in range(10):
            # Predict
            pred = model.forward(data)

            # Compare
            loss = criterion.forward(pred, target)

            # Learn
            loss.backward(Tensor(np.ones_like(loss.data)))
            optim.step()
            print(loss)

    @timeit("Test 7")
    def test7(self):
        # 8. Реализация сети с кросс-энтропией
        np.random.seed(0)
        # Исходные индексы
        data = Tensor(np.array([1, 2, 1, 2]), autograd=True)

        # Целевые индексы
        target = Tensor(np.array([0, 1, 0, 1]), autograd=True)

        model = Sequential([Embedding(3, 3), Tanh(), Linear(3, 4)])
        criterion = CrossEntropyLoss()

        optim = SGD(parameters=model.get_parameters(), alpha=0.1)
        print('CrossEntropyLoss')
        for i in range(10):
            # Прогноз
            pred = model.forward(data)

            # Сравнение
            loss = criterion.forward(pred, target)

            # Обучение
            loss.backward(Tensor(np.ones_like(loss.data)))
            optim.step()
            print(loss)


if __name__ == "__main__":
    unittest.main()
