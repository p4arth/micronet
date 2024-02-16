import numpy as np
from base import NN

class Relu():
    def __init__(self):
        self.grad = 0.0
    
    def __repr__(self):
        return "Relu()"
    
    def _backward(self, global_grad):
        print()
        print(self.__repr__())
        self.grad_mask = np.ones(self.z.shape, dtype = np.float32)
        self.grad = global_grad * self.grad_mask
        print("Grad shape = ", self.grad.shape)
        print()
        return self.grad

    def __call__(self, z):
        self.z = np.maximum(z.data, 0)
        return NN(data = self.z, grad_fn = self._backward)