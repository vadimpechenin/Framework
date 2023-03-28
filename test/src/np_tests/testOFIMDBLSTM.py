import unittest

import numpy as np
import classes.supportFunctions as SP
from classes.loadDataIMDB import load_dataIMDB
from model.modelLSTMForClassification import LSTMClassificationOfExpressions
from framework.tensor.tensor import Tensor
from testUtils import TestUtils
import pathlib


class LSTMModelForClassificationTest(unittest.TestCase):
    # Тестирование модели RNN
    def test(self):
        epochs = 10
        path = TestUtils.getMainResourcesIMDBFolder()
        filename = pathlib.Path(path).joinpath("reviews2.txt").resolve()
        fileNameLabels = pathlib.Path(path).joinpath("labels.txt").resolve()
        vocab, data, targetDataset = load_dataIMDB(filename, fileNameLabels, rawLength=200, sentenseLength=100)
        data = data/len(vocab)
        model = LSTMClassificationOfExpressions(vocab,data.shape[1])

        nTest = 0.2
        dataTest = data[data.shape[0]-round(data.shape[0]*nTest):]
        dataTrain = data[0:data.shape[0]-round(data.shape[0]*nTest)]
        targetDatasetTest, targetDatasetTrain = targetDataset[len(targetDataset)-round(len(targetDataset)*nTest):], \
            targetDataset[0:len(targetDataset)-round(len(targetDataset)*nTest)]
        batch_size_test = 40
        batch_size_train = 160
        # Обучение
        for iter in range(epochs):
            batch_size = 1
            total_loss = 0
            hidden = model.LSTM[0].init_hidden(batch_size=batch_size)
            #output_all = Tensor(np.zeros((model.embed.dim, len(targetDatasetTrain))), autograd=True)
            for i in range(dataTrain.shape[0]):
                t = dataTrain[i].reshape(1,dataTrain[i].shape[0])
                input = Tensor(t, autograd=True)
                output, hidden = model.LSTM[0].forward(input=input, hidden=hidden)
                output = model.model.forward(input=output)
                loss = model.criterion.forward(output, Tensor(targetDatasetTrain[i], autograd=True))
                loss.backward()
                model.optim.step()
                total_loss += loss.data
            if (iter % 1 == 0):
                print("Loss:", total_loss / (dataTrain.shape[0] / batch_size))
                SP.calculateResultsOfClassification(iter, model, dataTrain, dataTest, targetDatasetTest,
                                                    targetDatasetTrain, batch_size_test, batch_size_train)

        # Тестирование и проверка результатов
        SP.calculateResultsOfClassification(iter, model, dataTrain, dataTest, targetDatasetTest,
                                            targetDatasetTrain, batch_size_test, batch_size_train)
        g = 0

if __name__ == "__main__":
    unittest.main()
