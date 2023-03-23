import unittest

import numpy as np

from classes.loadDataIMDB import load_dataIMDB
from model.modelLinearForClassification import LinearClassificationOfExpressions
from framework.tensor.tensor import Tensor
from testUtils import TestUtils
import pathlib


class LinearModelForClassificationTest(unittest.TestCase):
    # Тестирование модели RNN
    def test(self):
        path = TestUtils.getMainResourcesIMDBFolder()
        filename = pathlib.Path(path).joinpath("reviews2.txt").resolve()
        fileNameLabels = pathlib.Path(path).joinpath("labels.txt").resolve()
        vocab, data, targetDataset = load_dataIMDB(filename, fileNameLabels, rawLength=200, sentenseLength=100)
        model = LinearClassificationOfExpressions(vocab,data.shape[1])

        nTest = 0.2
        dataTest = data[data.shape[0]-round(data.shape[0]*nTest):]
        dataTrain = data[0:data.shape[0]-round(data.shape[0]*nTest)]
        targetDatasetTest, targetDatasetTrain = targetDataset[len(targetDataset)-round(len(targetDataset)*nTest):], targetDataset[0:len(targetDataset)-round(len(targetDataset)*nTest)]
        target = Tensor(targetDatasetTrain, autograd=True)
        # Обучение
        for iter in range(1000):
            batch_size = 1
            total_loss = 0
            #output_all = Tensor(np.zeros((model.embed.dim, len(targetDatasetTrain))), autograd=True)
            for i in range(dataTrain.shape[0]):
                t = dataTrain[i].reshape(1,dataTrain[i].shape[0])
                input = Tensor(t, autograd=True)
                output = model.model.forward(input=input)
                loss = model.criterion.forward(output, Tensor(targetDatasetTrain[i], autograd=True))
                loss.backward()
                model.optim.step()
                total_loss += loss.data
            if (iter % 1 == 0):
                print("Loss:", total_loss / (dataTrain.shape[0] / batch_size))

        # Тест обученной сети
        batch_size = 1
        hidden = model.model.init_hidden(batch_size=batch_size)
        for t in range(5):
            input = Tensor(data[0:batch_size, t], autograd=True)
            rnn_input = model.embed.forward(input=input)
            output, hidden = model.model.forward(input=rnn_input, hidden=hidden)

        target = Tensor(data[0:batch_size, t + 1], autograd=True)
        loss = model.criterion.forward(output, target)

        ctx = ""
        for idx in data[0:batch_size][0][0:-1]:
            ctx += vocab[idx] + " "
        print("Context:", ctx)
        print("True:", vocab[target.data[0]])
        print("Pred:", vocab[output.data.argmax()])


if __name__ == "__main__":
    unittest.main()
