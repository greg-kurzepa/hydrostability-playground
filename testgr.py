import numpy as np

def f(x):
    return -(x-7)**2 + 3

def iterate(f, x_0=0):
    x = x_0
    gradient12 = 1
    count = 0

    dx = 1e-5
    threshold = 1e-5
    
    while abs(gradient12) > threshold:
        x1, x2, x3 = f(x), f(x+dx), f(x+2*dx)

        gradient12 = (x2 - x1) / dx
        gradient23 = (x3 - x2) / dx
        ggradient = (gradient23 - gradient12) / dx

        x -= gradient12 / ggradient
        
        count += 1
        if count > 1000:
                raise ValueError("Max number of iterations (1000) reached when calculating theta.")
        
    return x

print(iterate(f))