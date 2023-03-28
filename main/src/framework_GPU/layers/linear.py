# линейный слой
import cupy as cp
from framework_GPU.layers.layer import Layer
from framework_GPU.tensor.tensor import Tensor

class Linear(Layer):

    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()
        self.use_bias = bias

        W = cp.random.randn(n_inputs, n_outputs) * cp.sqrt(2.0 / (n_inputs))
        self.weight = Tensor(W, autograd=True)
        if (self.use_bias):
            self.bias = Tensor(cp.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)

        if (self.use_bias):
            self.parameters.append(self.bias)

    def forward(self, input):
        if (self.use_bias):
            return input.mm(self.weight) + self.bias.expand(0, len(input.data))
        return input.mm(self.weight)