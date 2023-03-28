import unittest

import numpy as np


from framework.tensor.tensor import Tensor

class TensorTest(unittest.TestCase):
    # Тестирование класса тензор, реализованного на cupy
    def test(self):
        x = Tensor([1, 2, 3, 4, 5])
        print(x)

        y = Tensor([2, 2, 2, 2, 2])
        print(y)

        z = x + y
        z.backward(Tensor(np.array([1, 1, 1, 1, 1])))
        print(z.creators)
        print(z.creation_op)

        a = Tensor([1, 2, 3, 4, 5], autograd=True)
        b = Tensor([2, 2, 2, 2, 2], autograd=True)
        c = Tensor([5, 4, 3, 2, 1], autograd=True)

        d = a + (-b)
        e = (-b) + c
        f = d + e

        f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
        print(b.grad.data == np.array([-2, -2, -2, -2, -2]))

        x = Tensor(np.array([[1, 2, 3],
                             [4, 5, 6]]))

        x.sum(0)

        x.sum(1)

        print(x.expand(dim=2, copies=4))

        print(x)

if __name__ == "__main__":
    unittest.main()
