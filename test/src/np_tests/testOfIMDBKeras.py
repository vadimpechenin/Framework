import unittest

import numpy as np

from classes.timeIt import timeit
from testUtils import TestUtils
import pathlib
import classes.supportFunctionsKeras as SP
import scipy.io

@timeit("LSTM Keras")
def train_test(pl_NN,maxlen,embedding_dims,num_neurons,x_train,y_train, batch_size,epochs,x_test, y_test):
    model = SP.build_LSTM_model(maxlen, embedding_dims, num_neurons)
    path = TestUtils.getSaveWeightsFolder()
    if (pl_NN == 1):
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test))

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
    # y_test_pedict = np.argmax(y_test_predict_)
    y_test_predict = (np.around(y_test_predict_)).astype('int32')
    return y_train_predict, y_test_predict, y_train_predict_, y_test_predict_

class IMDBClassificationKerasTest(unittest.TestCase):
    # Тестирование модели RNN для классификации IMDB набора
    def test(self):
        #Константы
        pl_NN = 1
        maxlen = 400 #Количество токенов
        num_neurons = 50  # Количество нейронов в ячейке памяти
        batch_size = 5  # Количество примеров, которые демонстрируются сети, перед обратным распространением ошибки и обновлением весов
        embedding_dims = 300  # Длины создаваемых для передачи в сверточную сеть векторов токенов
        epochs = 1
        #Путь к данным (уже использованы преобразования слов в векторы)
        path = TestUtils.getIMDBWithVectorsFolderFull()
        try:
            mat = scipy.io.loadmat(str(pathlib.Path(path).joinpath("data_test_train.mat").resolve()))
        except:
            path = TestUtils.getIMDBWithVectorsFolderHome()
            mat = scipy.io.loadmat(str(pathlib.Path(path).joinpath("data_test_train.mat").resolve()))
        x_train = np.array(mat['x_train'])
        x_test = np.array(mat['x_test'])
        y_train = np.array(mat['y_train']).reshape((x_train.shape[0],))
        y_test = np.array(mat['y_test']).reshape((x_test.shape[0],))
        #
        # Формируем однослойнуюмерную LSTM
        y_train_predict, y_test_predict, y_train_predict_, y_test_predict_ = train_test(pl_NN, maxlen, embedding_dims, num_neurons, x_train, y_train, batch_size, epochs, x_test, y_test)

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
