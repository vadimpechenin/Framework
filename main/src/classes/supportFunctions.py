"""
Вспомогательные функции для презентации результатов
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM,Flatten


def calculate_catecorical_crossentropy(yTarget, yCalculate):
    #Функция для расчета категориальной (в случае двух бинарной) кросс-энтропии
    #Количество значений
    N = yTarget.shape[0]
    #Корректировка yCalculate
    for j in range(yCalculate.shape[0]):
        if yCalculate[j]<1/10000:
            yCalculate[j] = 1/10000
        if yCalculate[j] > 1-1 / 10000:
            yCalculate[j] = 1-1 / 10000
    y = np.log2(yCalculate)
    y_neg = np.log2(1-yCalculate)
    y_mult = yTarget * y
    y_mult_neg = (1-yTarget) * y_neg
    matrix = np.array([y_mult,y_mult_neg])
    sum_ = np.sum(matrix,axis = 0)#
    result = -np.sum(sum_,axis = 0)/N
    return result

def build_LSTM_model(maxlen,embedding_dims,num_neurons):
    #Компиляция однослойной реккурентной сети с долгосрочной памятью LSTM
    #maxlen = 400
    #embedding_dims = 300  # Длины создаваемых для передачи в сверточную сеть векторов токенов
    #num_neurons = 50  # Количество нейронов в ячейке памяти
    #kernel_size = 3  # Ширина фильтров. Фактически фильтры будут каждый представлять собой матрицу весов размером embedding_dims x kernel_size, то есть 50ч3 в нашем случае
    #hidden_dims = 250  # Количество нейронов в плоской упреждающей нейронной сети в конце цепочки
    model = Sequential()
    #model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))

    model.add(LSTM(
        num_neurons,
        input_shape=(maxlen, embedding_dims),
        return_sequences = True
    ))#

    model.add(Dropout(0.2))
    model.add(Flatten())
    # Слои классификации, процеживание
    model.add(Dense(1, activation = 'sigmoid'))
    # Компиляция LSTN
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    return model