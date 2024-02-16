import numpy as np
from base import NN

class MSELoss():
    def __init__(self):
        self.grad = 0.0

    def _backward(self, global_grad):
        self.grad = np.array([-(2 / self.yhat.shape[0])])
        return (self.grad * global_grad)

    def __call__(self, y, yhat):
        assert y.shape[0] == yhat.shape[0]
        self.yhat = yhat
        out = (1 / y.shape[0]) * np.sum((y - yhat)**2)
        return NN(data = out, grad_fn = self._backward)