from microgradEngine import value
from neuralNetEngine import MLP

# ========!!EXAMPLE!!=========== #
x = [2.0, 3.0, -1.0]
n = MLP(2, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

ypred  = [n(xx) for xx in xs]

for k in range(100):
    # Forward pass
    ypred  = [n(x) for x in xs]
    loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), value(0))
    
    # Backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # Update rule
    for p in n.parameters():
        p.data += -0.001*p.grad  
        
    print(f'epoch #{k}: loss = {loss.data}')
