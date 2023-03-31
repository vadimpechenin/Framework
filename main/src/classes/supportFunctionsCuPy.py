"""
Вспомогательные функции для презентации результатов
"""
import cupy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM,Flatten

from framework_GPU.tensor.tensor import Tensor


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

def calculateResultsOfClassification(iteration, model, dataTrain, dataTest, targetDatasetTest,targetDatasetTrain,batch_size_test,batch_size_train):
    #batch_size_test = 40
    #batch_size_train = 160

    correct, score = calculateMetrics(batch_size_test,model,dataTest,targetDatasetTest)
    correctTrain, scoreTrain = calculateMetrics(batch_size_train, model, dataTrain, targetDatasetTrain)
    print("***********")
    print("Эпоха: " + str(iteration))
    print(
        'Точность тестирования: ' + str(correct / float(batch_size_test)) + 'Функция потерь (бинарная кроссэнтропия): ' + str(
            score))
    print('Точность обучения: ' + str(
        correctTrain / float(batch_size_train)) + 'Функция потерь (бинарная кроссэнтропия): ' + str(scoreTrain))

def calculateMetrics(batch_size,model,dataTest,targetDatasetTest):
    hidden = model.LSTM[0].init_hidden(batch_size=batch_size)
    input = Tensor(dataTest[0:batch_size, :], autograd=True)
    lstm_input_words = model.embed.forward(input=input)
    # lstm_input_sentence = TensorC(lstm_input_words.data.sum(1) / lstm_input_words.data.shape[2], autograd=True)
    lstm_input_sentence = Tensor(lstm_input_words.data.reshape(lstm_input_words.data.shape[0],
                                                                lstm_input_words.data.shape[1] *
                                                                lstm_input_words.data.shape[2]), autograd=True)
    output, hidden = model.LSTM[0].forward(input=lstm_input_sentence, hidden=hidden)
    y_test_predict_ = model.model.forward(input=output)
    y_test_predict = (np.around(y_test_predict_.data)).astype('int32')
    y_test = np.array(targetDatasetTest[0:batch_size])
    correct = 0
    for i in range(y_test_predict.shape[0]):
        if (np.abs(y_test[i] - y_test_predict[i]) < 0.5):
            correct += 1
    # Кросс-энтропия
    score = calculate_catecorical_crossentropy(y_test, y_test_predict_.data.reshape((dataTest.shape[0],)))
    return correct, score
