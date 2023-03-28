import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from testUtils import TestUtils
import pathlib

pl_exp = "article"
path = TestUtils.getResources()
data = pd.read_excel(str(pathlib.Path(path).joinpath("results_article1.xlsx").resolve()), index_col=0)
epochs = list(data.index)
acc = list(data.acc)
val_acc = list(data.val_acc)
loss = list(data.loss)
val_loss = list(data.val_loss)
#epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, acc, 'bo', label='Точность обучения')
plt.plot(epochs, val_acc, 'b', label='Точность теста')
# plt.title('Точность')
plt.legend()
plt.savefig(str(pathlib.Path(path).joinpath('results_acc_' + str(pl_exp) + '_.jpeg').resolve()), dpi=600)
plt.close()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Функция потерь обучения')
plt.plot(epochs, val_loss, 'b', label='Функция потерь теста')
# plt.title('Потери')
plt.legend()
plt.savefig(str(pathlib.Path(path).joinpath('results_loss_' + str(pl_exp) + '_.jpeg').resolve()), dpi=600)
plt.close()