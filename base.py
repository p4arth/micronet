class NN():
    def __init__(self, data, grad_fn = None):
        self.grad_fn = grad_fn
        self.data = data