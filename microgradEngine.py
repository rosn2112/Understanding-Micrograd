import numpy as np

class value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children) # we want to keep all the values of all the operations
        self._op = _op # we want to keep all the operations of all the operations
        self.label = label # label of operation 

    def __repr__(self):
        return f'value(data = {self.data})' # representation
    
    def __add__(self, other):
        other = other if isinstance(other, value) else value(other) # a + 8 is valid but 8 + a is invalid; it is here to fix that
        out = value(self.data + other.data, (self, other), '+')

        def _backward(): # back prop
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, value) else value(other) # a * 8 is valid but 8 * a is invalid; it is here to fix that
        out = value(self.data * other.data, (self, other), '*')

        def _backward(): # back prop
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "supports int/float for now"
        out = value(self.data**other, (self,), f'**{other}')

        def _backward(): # back prop
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other): 
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * other**-1

    def exp(self):
        x = self.data
        out = value(np.exp(x), (self, ), 'exp')

        def _backward(): # back prop
            self.grad = out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        n = self.data
        t = (np.exp(2*n) - 1)/(np.exp(2*n) + 1)
        out = value(t, (self, ), 'tanh')

        def _backward(): # back prop
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def backward(self): # back prop
        topo = [] # topological sort
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()