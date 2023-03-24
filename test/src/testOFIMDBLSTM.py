import unittest

import numpy as np
import classes.supportFunctions as SP
from classes.loadDataIMDB import load_dataIMDB
from model.modelLSTMForClassification import LSTMClassificationOfExpressions
from framework.tensor.tensor import Tensor
from testUtils import TestUtils
import pathlib


class LinearModelForClassificationTest(unittest.TestCase):
    # Тестирование модели RNN
    def test(self):
        epochs = 10
        path = TestUtils.getMainResourcesIMDBFolder()
        filename = pathlib.Path(path).joinpath("reviews2.txt").resolve()
        fileNameLabels = pathlib.Path(path).joinpath("labels.txt").resolve()
        vocab, data, targetDataset = load_dataIMDB(filename, fileNameLabels, rawLength=200, sentenseLength=100)
        model = LSTMClassificationOfExpressions(vocab,data.shape[1])

        nTest = 0.2
        dataTest = data[data.shape[0]-round(data.shape[0]*nTest):]
        dataTrain = data[0:data.shape[0]-round(data.shape[0]*nTest)]
        targetDatasetTest, targetDatasetTrain = targetDataset[len(targetDataset)-round(len(targetDataset)*nTest):], targetDataset[0:len(targetDataset)-round(len(targetDataset)*nTest)]
        target = Tensor(targetDatasetTrain, autograd=True)
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

        # Тестирование и проверка результатов
        batch_size = 40
        hidden = model.LSTM[0].init_hidden(batch_size=batch_size)
        input = Tensor(dataTest[0:batch_size, :], autograd=True)
        output, hidden = model.LSTM[0].forward(input=input, hidden=hidden)
        y_test_predict_ = model.model.forward(input=output)
        y_test = np.array(targetDatasetTest[0:batch_size])

        batch_size = 160
        hidden = model.LSTM[0].init_hidden(batch_size=batch_size)
        input = Tensor(dataTrain[0:batch_size, :], autograd=True)
        output, hidden = model.LSTM[0].forward(input=input, hidden=hidden)
        y_train_predict_ = model.model.forward(input=output)

        y_train = np.array(targetDatasetTrain[0:batch_size])

        y_train_predict = (np.around(y_train_predict_.data)).astype('int32')

        y_test_predict = (np.around(y_test_predict_.data)).astype('int32')
        correct = 0
        correctTrain = 0
        for i in range(y_train_predict.shape[0]):
            if (i < y_test_predict.shape[0]):
                if (np.abs(y_test[i] - y_test_predict[i]) < 0.5):
                    correct += 1
            if (np.abs(y_train[i] - y_train_predict[i]) < 0.5):
                correctTrain += 1
        total = y_test_predict.shape[0]
        totalTrain = y_train_predict.shape[0]
        # Кросс-энтропия
        scoreTrain = SP.calculate_catecorical_crossentropy(y_train, y_train_predict_.data.reshape((dataTrain.shape[0],)))
        score = SP.calculate_catecorical_crossentropy(y_test, y_test_predict_.data.reshape((dataTest.shape[0],)))
        print(
            'Точность тестирования: ' + str(correct / float(total)) + 'Функция потерь (бинарная кроссэнтропия): ' + str(
                score))
        print('Точность обучения: ' + str(
            correctTrain / float(totalTrain)) + 'Функция потерь (бинарная кроссэнтропия): ' + str(scoreTrain))
        g = 0

if __name__ == "__main__":
    unittest.main()
