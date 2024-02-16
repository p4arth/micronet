import numpy as np
from activations import Relu
from losses import MSELoss
from base import NN
from linear import Linear
np.random.seed(42)

def build_model(input_features):
    modules = [
        Linear(input_features, 2),
        Relu(),
        Linear(2, 2),
        Relu(),
        Linear(2, 1),
        Relu(),
    ]
    return modules

def forward(modules, x):
    back_graph = []
    for m in modules:
        # print(x.data.shape)
        x = m(x)
        
        back_graph.append(x)
    return back_graph, x

def backward(back_graph, loss):
    global_grad = loss.grad_fn(1.0)
    for m in reversed(back_graph):
        global_grad = m.grad_fn(global_grad)
    return global_grad

def step(modules): 
    for m in modules:
        if hasattr(m, "w"):
            m.w = m.w - (LR * m.grad)

def zero_grad(modules):
    for m in modules:
        m.grad = 0.0

# Initializations.
NFEATURES = 3
EPOCHS = 500
LR = 0.0001
x1 = NN(data = np.random.randn(1, NFEATURES))
y = np.array([1.0])
criterion = MSELoss()
layers = build_model(input_features = NFEATURES)


for epoch in range(EPOCHS):
    print(f"Epoch[{epoch}]")
    zero_grad(layers)
    back_graph, out = forward(layers, x1)
    loss = criterion(y, out.data.squeeze(0))
    backward(back_graph, loss)
    step(layers)
print("Loss = ", loss.data)