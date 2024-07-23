import random
from microgradEngine import value

class neuron:
    def __init__(self, nIn):
        self.w = [value(random.uniform(-1, 1)) for _ in range(nIn)]
        self.b = value(random.uniform(-1, 1))
    
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class layer:
    def __init__(self, nIn, nOut):
        self.neurons = [neuron(nIn) for _ in range(nOut)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nIn, nOuts):
        sz = [nIn] + nOuts
        self.layers = [layer(sz[i], sz[i+1]) for i in range(len(nOuts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]