from framework.layers.crossentropyloss import CrossEntropyLoss
from framework.layers.embedding import Embedding
from framework.layers.lstmcell import LSTMCell
from framework.optimization.sgd import SGD


class ModelLSTM():
    #Модель нейронной сети с LSTM, краткосрочной памятью
    def __init__(self,vocab):
        self.embed = Embedding(vocab_size=len(vocab),dim=512)
        self.model = LSTMCell(n_inputs=512, n_hidden=512, n_output=len(vocab))
        #Это немного поможет в обучении
        self.model.w_ho.weight.data *= 0

        self.criterion = CrossEntropyLoss()
        self.optim = SGD(parameters=self.model.get_parameters() + self.embed.get_parameters(), alpha=0.05)
