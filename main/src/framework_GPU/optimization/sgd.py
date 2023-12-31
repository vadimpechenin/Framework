#Класс, реализующий оптимизатор стохастического градиентного спуска

class SGD(object):

    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):

        for p in self.parameters:
            try:
                p.data -= p.grad.data * self.alpha
            except:
                p.data = 0
            if(zero):
                p.grad.data *= 0