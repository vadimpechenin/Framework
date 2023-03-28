from framework_GPU.layers.layer import Layer


class Dropout(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input, keep_prob = 0.5):
        return input.dropout(keep_prob)