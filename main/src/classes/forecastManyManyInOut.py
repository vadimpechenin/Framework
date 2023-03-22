"""
Класс для реализации всех методов со страницы 120
"""

import numpy as np
import matplotlib.pyplot as plt

class ForecastManyManyInOut:

    @staticmethod
    def w_sum(a, b):
        assert (len(a) == len(b))
        output = 0
        for i in range(len(a)):
            output += (a[i] * b[i])
        return output

    @staticmethod
    def vect_mat_mul(vect, matrix):
        assert (len(vect)) == len(matrix)
        output = [0.0, 0.0, 0.0]
        for i in range(len(vect)):
            output[i] = ForecastManyManyInOut.w_sum(vect, matrix[i])
        return output

    @staticmethod
    def neural_network(input, weights):
        pred = ForecastManyManyInOut.vect_mat_mul(input, weights)
        return pred

    @staticmethod
    def ele_mul(number, vector):
        output = [0, 0, 0]
        assert (len(output) == len(vector))

        for i in range(len(vector)):
            output[i] = number * vector[i]

        return output

    @staticmethod
    def outer_prod(vec_a, vec_b):
        out = np.zeros((len(vec_a), len(vec_b)))
        for i in range(len(vec_a)):
            for j in range(len(vec_b)):
                out[i][j] = vec_a[i] * vec_b[j]
        return out

    @staticmethod
    def plot_errors(epoch, error,name):
        #Метод для графической визуализации ошибок по эпохам обучения
        fig = plt.figure(figsize=(15,7))
        ax = [None] * len(error[1])
        arrayEpoch = [i for i in range(epoch)]
        for iter in range(len(error[1])):
            y = []
            for errorVector in error:
                y.append(errorVector[iter])
            # только одна панель для рисования графиков
            ax[iter] = fig.add_subplot(1, len(error[1]), iter+1)
            ax[iter].plot(arrayEpoch, y, 'k')
            # график косинуса:
            # подпись по оси x
            ax[iter].set_xlabel('Эпоха')
            # подпись по оси y
            ax[iter].set_ylabel('Ошибка')
            # заголовок рисунка
            ax[iter].set_title(str(iter))
            ax[iter].grid()
        fig.show()
        fig.savefig(name + ".png", orientation='landscape', dpi=300)

    @staticmethod
    def plotGradients(epoch, activation, gradient, name):
        # Метод для графической визуализации градиентов и значений функций активации
        fig = plt.figure(figsize=(15, 7))
        ax = [None] * len(activation[1]) * 2
        arrayEpoch = [i for i in range(epoch)]
        for iter in range(len(activation[1])*2):
            y = []
            if iter<2:
                error = activation
                for errorVector in error:
                    y.append(errorVector[iter])
            else:
                error = gradient
                iter=iter-2
                for errorVector in error:
                    y.append(errorVector[iter])
                iter = iter + 2

            # только одна панель для рисования графиков
            ax[iter] = fig.add_subplot(2, len(error[1]), iter + 1)
            ax[iter].plot(arrayEpoch, y, 'k')
            # график косинуса:
            # подпись по оси x
            ax[iter].set_xlabel('Эпоха')
            # подпись по оси y
            if iter < 2:
                ax[iter].set_ylabel('Расчет')
            else:
                ax[iter].set_ylabel('Градиент')
            # заголовок рисунка
            ax[iter].set_title(str(iter))
            ax[iter].grid()
        fig.show()
        fig.savefig(name + ".png", orientation='landscape', dpi=300)

    @staticmethod
    def plot_accuracy(epoch, acc,name):
        #Метод для графической визуализации ошибок по эпохам обучения
        fig = plt.figure(figsize=(15,7))
        ax = [None] * 1 #len(acc)
        arrayEpoch = [i for i in range(epoch)]
        for iter in range(1):
            #y = []
            #for errorVector in acc:
            #    y.append(errorVector[iter])
            # только одна панель для рисования графиков
            ax[iter] = fig.add_subplot(1, 1, iter+1)
            ax[iter].plot(arrayEpoch, acc, 'k')
            # график косинуса:
            # подпись по оси x
            ax[iter].set_xlabel('Эпоха')
            # подпись по оси y
            ax[iter].set_ylabel('Точность')
            # заголовок рисунка
            ax[iter].set_title(str(iter))
            ax[iter].grid()
        fig.show()
        fig.savefig(name + ".png", orientation='landscape', dpi=300)
