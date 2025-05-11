# Imports
import numpy as np
import matplotlib.pyplot as plt
from modelos_RNA.adaline import Adaline
from config import Config

plot_graph = True

# Leitura dos dados
aerogerador = np.loadtxt("trabalhos\\Trabalho_Computacional_AV2\\datasets\\aerogerador.dat")

# Regressão Linear Simples

# ADALINE
# Instanciação do modelo
adaline = Adaline(η=0.0001, ϵ=0.0001, epochs=1000)

# Separação da variável dependente e independente
X = aerogerador[:, 0]  # Variável independente (primeira coluna)
y = aerogerador[:, 1]   # Variável dependente (segunda coluna)

plt.figure(1)
plt.scatter(X, y, color='blue', label='Dados Originais')

# Normalização dos dados
X = (X - np.mean(X)) / np.std(X)
y = (y - np.mean(y)) / np.std(y)
plt.figure(2)
plt.scatter(X, y, color='red', label='Dados Normalizados')

# Separação dos dados em treino e teste
train_size = int(Config.REGRESSION_TRAIN * len(X))  # 80% para treino
# 20% para teste
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Etapa de treinamento
adaline.fit(X_train.reshape(-1, 1), y_train)
plt.figure(3)
adaline.plot_EQMs()
if plot_graph:
    plt.show()
predictions = adaline.predict_regression(X_test.reshape(-1, 1))
print(adaline.__EQMs)
# MSE (Erro Quadrático Médio)
EQM = np.mean((y_test - predictions) ** 2)
# R² (coeficiente de determinação)
ss_res = np.sum((y_test - predictions) ** 2)              # Soma dos quadrados dos resíduos
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)          # Soma total dos quadrados
r2 = 1 - (ss_res / ss_tot)
print(f"EQM: {EQM:.4f}")
print(f"R²: {r2:.4f}")