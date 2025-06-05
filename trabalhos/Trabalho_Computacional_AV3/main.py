import numpy as np
from algoritmos.genetico import GeneticAlgorithm

def f(x1, x2):
    return np.abs(x1*x2*np.sin(x2*np.pi/4) + 1)

genetic = GeneticAlgorithm(
    f=f,
    N=10,
    p=2,
    min=-2,
    max=20,
    mutation_probability=0.02,
    n_bits=5,
    max_generations=100,
    patience=20
)

x, y = genetic.start()
print(f"x: {x}, y: {y}")