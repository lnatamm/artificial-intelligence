from algoritmos.otimizacao import *
from algoritmos.genetico import GeneticAlgorithm
import numpy as np
import matplotlib.pyplot as plt

# def f(x1, x2):
#     return np.abs(x1*x2*np.sin(x2*np.pi/4) + 1)
# def f(x1, x2):
#     return x1**2 + x2**2
# def f(x1, x2):
#     return -np.sin(x1) * np.sin((x1**2 / np.pi)**20) - np.sin(x2) * np.sin(((2 * x2**2) / np.pi)**20)
# def plot_function(f, x_min, x_max, y_min=None, y_max=None):
#     if y_min is None:
#         y_min = x_min
#     if y_max is None:
#         y_max = x_max
#     figure = plt.figure()
#     ax = figure.add_subplot(projection='3d')
#     x_axis = np.linspace(x_min, x_max, 100)
#     if y_min is None and y_max is None:
#         y_axis = x_axis
#     else:
#         y_axis = np.linspace(y_min, y_max, 100)
#     mesh_x, mesh_y = np.meshgrid(x_axis, y_axis)
#     ax.plot_surface(mesh_x, mesh_y, f(mesh_x, mesh_y), cmap='viridis')
# plot_function(f, 0, 3)
# plt.show()

# def numpy_mode(data):

#     # Index and counts of all elements in the array
#     (sorted_data, idx, counts) = np.unique(data, return_index=True, return_counts=True)

#     # Index of element with highest count (i.e. the mode)
#     index = idx[np.argmax(counts)]

#     # Return the element with the highest count
#     return data[index]

# restricoes = np.array([[-100, 100], [-100, 100]])
# x_axis = np.linspace(np.min(restricoes),np.max(restricoes),500)
# X,Y = np.meshgrid(x_axis,x_axis)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(X,Y,f(X,Y),rstride=20,cstride=20,cmap='gray',alpha=.4)
# rodada = 0
# resultados = []
# while rodada < 250:
#     lrs = LocalRandomSearch(max_it=1000, sigma=1, func=f, restricoes=np.array([[-100, 100], [-100, 100]]))
#     resultados.append(lrs.search())
#     print(f"Rodada {rodada+1}: {resultados[-1]}")
#     rodada += 1
# numpy_mode(resultados)

# hill_climbing = HillClimbing(
#     max_it=1000,
#     epsilon=1,
#     func=f,
#     max_viz=50,
#     restricoes=np.array([[-2, 20], [-2, 20]])
# )
# global_search = GlobalRandomSearch(
#     max_it=1000,
#     func=f,
#     restricoes=np.array([[-2, 20], [-2, 20]])
# )

# hill_climbing.search()

# local_search.search()
# genetic = GeneticAlgorithm(
#     f=f,
#     N=10,
#     p=2,
#     min=-2,
#     max=20,
#     mutation_probability=0.02,
#     n_bits=5,
#     max_generations=100,
#     patience=20
# )

# x, y = genetic.start()
# print(f"x: {x}, y: {y}")

def plot_function(f, x_min, x_max, y_min=None, y_max=None):
    if y_min is None:
        y_min = x_min
    if y_max is None:
        y_max = x_max
    figure = plt.figure()
    ax = figure.add_subplot(projection='3d')
    x_axis = np.linspace(x_min, x_max, 100)
    if y_min is None and y_max is None:
        y_axis = x_axis
    else:
        y_axis = np.linspace(y_min, y_max, 100)
    mesh_x, mesh_y = np.meshgrid(x_axis, y_axis)
    ax.plot_surface(mesh_x, mesh_y, f(mesh_x, mesh_y), cmap='viridis')
    
def numpy_mode(data):

    # Index and counts of all elements in the array
    (sorted_data, idx, counts) = np.unique(data, return_index=True, return_counts=True)

    # Index of element with highest count (i.e. the mode)
    index = idx[np.argmax(counts)]-1

    # Return the element with the highest count
    return data[index]

# 1
def f(x1, x2):
    return x1**2 + x2**2

restricoes = np.array([[-100, 100], [-100, 100]])
x_axis = np.linspace(np.min(restricoes),np.max(restricoes),500)
X,Y = np.meshgrid(x_axis,x_axis)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X,Y,f(X,Y),rstride=20,cstride=20,cmap='viridis',alpha=.4)
rodada = 0
variaveis = []
resultados = []
while rodada < 500:
    hill_climbing = HillClimbing(max_it=1000,epsilon=100,func=f,max_viz=100, restricoes=restricoes, max=False)
    variavel, resultado = hill_climbing.search()
    variaveis.append(variavel)
    resultados.append(resultado)
    
    rodada += 1
resultados_arredondados = np.round(resultados, 5)
mode = numpy_mode(resultados_arredondados)
variaveis_dos_resultados = []
for i, resultado in enumerate(resultados_arredondados):
    if resultado == mode:
        variaveis_dos_resultados.append(variaveis[i])
print(f'Moda: {mode}')
print(f'Variáveis que geram o resultado da moda: {variaveis_dos_resultados}')
ax.scatter(variaveis_dos_resultados[0][0], variaveis_dos_resultados[0][1], f(variaveis_dos_resultados[0][0], variaveis_dos_resultados[0][1]), c='red',marker='*')