import unittest

import numpy as np

from classes.loadData import load_data
from model.modelRNN import ModelRNN
from framework.tensor.tensor import Tensor
from testUtils import TestUtils
import pathlib

class LoadModelRNNTest(unittest.TestCase):
    # Тестирование модели RNN
    def test(self):
        path = TestUtils.getMainResourcesFolder()
        filename = pathlib.Path(path).joinpath("qa1_single-supporting-fact_train.txt").resolve()
        data, vocab = load_data(filename)
        model = ModelRNN(vocab)

        # Обучение
        for iter in range(1000):
            batch_size = 100
            total_loss = 0

            hidden = model.model.init_hidden(batch_size=batch_size)

            for t in range(5):
                input = Tensor(data[0:batch_size, t], autograd=True)
                rnn_input = model.embed.forward(input=input)
                output, hidden = model.model.forward(input=rnn_input, hidden=hidden)

            target = Tensor(data[0:batch_size, t + 1], autograd=True)
            loss = model.criterion.forward(output, target)
            loss.backward()
            model.optim.step()
            total_loss += loss.data
            if (iter % 200 == 0):
                p_correct = (target.data == np.argmax(output.data, axis=1)).mean()
                print("Loss:", total_loss / (len(data) / batch_size), "% Correct:", p_correct)

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
