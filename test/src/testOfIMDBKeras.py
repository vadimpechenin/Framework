import unittest

import numpy as np

from classes.loadDataIMDB import load_dataIMDB
from model.modelLinear import LinearClassification
from framework.tensor.tensor import Tensor
from testUtils import TestUtils
import pathlib
import classes.supportFunctions as SP
import scipy.io

class IMDBClassificationKerasTest(unittest.TestCase):
    # Тестирование модели LSTM для классификации с использованием tensorflow
    def test(self):
        #Константы
        pl_NN = 1
        maxlen = 400 #Количество токенов
        num_neurons = 50  # Количество нейронов в ячейке памяти
        batch_size = 32  # Количество примеров, которые демонстрируются сети, перед обратным распространением ошибки и обновлением весов
        embedding_dims = 300  # Длины создаваемых для передачи в сверточную сеть векторов токенов
        epochs = 2
        #Путь к данным (уже использованы преобразования слов в векторы)
        path = TestUtils.getIMDBWithVectorsFolder()
        mat = scipy.io.loadmat(str(pathlib.Path(path).joinpath("data_test.mat").resolve()))
        mat1 = scipy.io.loadmat(str(pathlib.Path(path).joinpath("data_train1.mat").resolve()))
        if (1==0):
            mat2 = scipy.io.loadmat(str(pathlib.Path(path).joinpath("data_train2.mat").resolve()))
            mat3 = scipy.io.loadmat(str(pathlib.Path(path).joinpath("data_train3.mat").resolve()))  # mat73.loadmat
            x_train1 = np.array(mat1['x_train'])
            x_train2 = np.array(mat2['x_train'])
            x_train3 = np.array(mat3['x_train'])
            x_train = np.vstack([x_train1, x_train2])
            x_train = np.vstack([x_train, x_train3])
        else:
            x_train = np.array(mat1['x_train'])
        x_test = np.array(mat['x_test'])
        y_train = np.array(mat['y_train'])
        if (1==1):
            t = np.linspace(int(x_train.shape[0]), int(x_train.shape[0]*3), int(x_train.shape[0]*2)+1).astype('int32')
            y_train = y_train.reshape((x_train.shape[0]*3+1,))
            y_train= np.delete(y_train, t)
            y_test = np.array(mat['y_test']).reshape((x_test.shape[0],))
            t = np.linspace(0, 1899, 1900).astype('int32')
            y_test = np.delete(y_test, t)
            x_test = np.delete(x_test, t, axis=0)
            t = np.linspace(0, 2332, 2333).astype('int32')
            x_train = np.delete(x_train, t, axis=0)
            y_train = np.delete(y_train, t)
        else:
            t = np.linspace(0, int(y_train.shape[0] / 3) - 1, int(y_train.shape[0] / 3)).astype('int32')
            x_train1 = np.delete(x_train1, t)
            x_train2 = np.delete(x_train2, t)
            x_train3 = np.delete(x_train3, t)
            y_train = y_train.reshape((x_train.shape[0],))
            y_test = np.array(mat['y_test']).reshape((x_test.shape[0],))
        #
        # Формируем однослойнуюмерную LSTM
        model = SP.build_LSTM_model(maxlen, embedding_dims, num_neurons)
        path = TestUtils.getSaveWeightsFolder()
        if (pl_NN==1):
            history = model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs = epochs,
                      validation_data=(x_test,y_test))

            model_structure = model.to_json()
            with open(str(pathlib.Path(path).joinpath("lstm_model.json").resolve()), "w") as json_file:
                json_file.write(model_structure)
            model.save_weights(str(pathlib.Path(path).joinpath("lstm_weights.h5").resolve()))
        else:
            # Загрузка сохраненнной модели
            from keras.models import model_from_json

            with open(str(pathlib.Path(path).joinpath("lstm_model.json").resolve()), "r") as json_file:
                json_string = json_file.read()
            model = model_from_json(json_string)
            model.load_weights(str(pathlib.Path(path).joinpath("lstm_weights.h5").resolve()))

        y_train_predict_ = model.predict(x_train)
        y_train_predict = (np.around(y_train_predict_)).astype('int32')

        y_test_predict_ = model.predict(x_test)
        #y_test_pedict = np.argmax(y_test_predict_)
        y_test_predict = (np.around(y_test_predict_)).astype('int32')
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
        scoreTrain = SP.calculate_catecorical_crossentropy(y_train, y_train_predict_.reshape((x_train.shape[0],)))
        score = SP.calculate_catecorical_crossentropy(y_test, y_test_predict_.reshape((x_test.shape[0],)))
        print(
            'Точность тестирования: ' + str(correct / float(total)) + 'Функция потерь (бинарная кроссэнтропия): ' + str(
                score))
        print('Точность обучения: ' + str(
            correctTrain / float(totalTrain)) + 'Функция потерь (бинарная кроссэнтропия): ' + str(scoreTrain))

print('Build model...')

if __name__ == "__main__":
    unittest.main()
