import numpy as np
from base import NN

class Linear():
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.w = np.random.randn(in_features, out_features)
        self.grad = 0.0
    
    def __repr__(self):
        return f"Linear(in_features = {self.in_features}, out_features = {self.out_features})"
    
    def _backward(self, h_prev_grad):
        print(self.__repr__())
        # # print(global_grad.shape)
        # print(global_grad.ndim)
        if (h_prev_grad.ndim == 1):
            h_prev_grad = np.expand_dims(h_prev_grad, -1)
        # print(h_prev_grad.shape, self.x.shape)
        self.grad = self.x.T @ h_prev_grad
        # self.grad = global_grad * self.x
        # print(self.w.shape)
        print(self.grad.shape)
        
        print("Grad shape =", self.grad.shape)
        assert self.w.shape == self.grad.shape, "Grad and weight shapes not matching"
        
        # print("linear_grad", self.grad)
        print()
        # print("grad",self.grad.shape)
        return h_prev_grad

    def __call__(self, x):
        self.x = x.data
        z = x.data @ self.w
        return NN(data = z, grad_fn = self._backward)