import unittest

import numpy as np

from classes.loadDataIMDB import load_dataIMDB
from model.modelLinear import LinearClassification
from framework.tensor.tensor import Tensor
from testUtils import TestUtils
import pathlib


class LinearModelTest(unittest.TestCase):
    # Тестирование модели RNN
    def test(self):
        data = Tensor(np.array([1,2,1,2]), autograd = True)
        target = Tensor(np.array([[0],[1],[0],[1]]), autograd = True)

        model = LinearClassification(1)

        # Обучение
        for i in range(10):
            output = model.model.forward(data)
            loss = model.criterion.forward(output, target)
            loss.backward()
            model.optim.step()
            print(loss)
        g = 0
        data_d = np.array([[0,0], [0,1], [1,0], [1,1]])
        target_d = np.array([[0], [1], [0], [1]])
        data = Tensor(data_d, autograd=True)
        target = Tensor(target_d, autograd=True)

        model = LinearClassification(0)
        # Обучение
        for i in range(10):
            output = model.model.forward(data)
            loss = model.criterion.forward(output, target)
            loss.backward()
            model.optim.step()
            print(loss)
        # Обучение
        for i in range(10):
            total_loss = 0
            for i in range(data_d.shape[0]):
                output = model.model.forward(Tensor(data_d[i], autograd=True))
                loss = model.criterion.forward(output, Tensor(target_d[i], autograd=True))
                loss.backward()
                model.optim.step()
                total_loss+=loss.data
            print(total_loss)

if __name__ == "__main__":
    unittest.main()
