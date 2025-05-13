# Imports
import numpy as np
import matplotlib.pyplot as plt
from modelos_RNA.adaline import Adaline
from modelos_RNA.multi_layer_perceptron import MLP
from config import Config

# Leitura dos dados
spiral = np.loadtxt("trabalhos\\Trabalho_Computacional_AV2\\datasets\\Spiral3d.csv", delimiter=",")

# Classificação Linear Simples
# Separação das variável dependente e independentes
X = spiral[:, :-1]  # Variável independente (todas as colunas menos a última)
y = spiral[:, -1]   # Variável dependente (última coluna)

if Config.CLASSIFICATION_PLOT_GRAPH:
    # Faz o plot 3d dos dados
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Dados Originais')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', label='Dados Originais')

# Normalização dos dados
X = (X - np.min(X)) / (np.max(X) - np.min(X))
y = (y - np.min(y)) / (np.max(y) - np.min(y))

if Config.CLASSIFICATION_PLOT_GRAPH:
    # Faz o plot 3d dos dados
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Dados Normalizados')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', label='Dados Normalizados')

plt.show()

# Instanciando os modelos
adaline = Adaline(η=Config.CLASSIFICATION_LEARNING_RATE, ϵ=Config.CLASSIFICATION_EPSILON, epochs=Config.CLASSIFICATION_EPOCHS)
mlp = MLP(input_size=3, q=[4, 4], m=2, η=Config.CLASSIFICATION_LEARNING_RATE, ϵ=Config.CLASSIFICATION_EPSILON, tolleration=10, epochs=Config.CLASSIFICATION_EPOCHS)

accuracies_adaline = []
accuracies_mlp = []
sensitivities_adaline = []
sensitivities_mlp = []
specificities_adaline = []
specificities_mlp = []

for round in range(Config.CLASSIFICATION_N_ROUNDS):
    print(f"Rodada: {round + 1}")
    # Aleatorização dos dados
    index = np.random.permutation(spiral.shape[0])
    X_shuffled = X[index]
    y_shuffled = y[index]
    # Separação dos dados em treino, teste e validação
    train_size = int(Config.CLASSIFICATION_TRAIN * len(X))
    validation_size = int(Config.CLASSIFICATION_VALIDATION * len(X))
    # 80% para treino
    X_train = X_shuffled[:train_size]
    X_test = X_shuffled[train_size:]
    # 20% para teste
    y_train = y_shuffled[:train_size]
    y_test = y_shuffled[train_size:]
    # 10% para validação
    X_validation = X_test[:validation_size]
    y_validation = y_test[:validation_size]
    # Etapa de treinamento
    # Adaline
    adaline.fit(X_train, y_train)
    predictions_adaline = adaline.predict_classification(X_test)
    # MLP
    # Codificando as classes
    y_train = [np.array([1, 0]) if i == 1 else np.array([0, 1]) for i in y_train]
    y_test = [np.array([1, 0]) if i == 1 else np.array([0, 1]) for i in y_test]
    y_validation = [np.array([1, 0]) if i == 1 else np.array([0, 1]) for i in y_validation]
    predictions_mlp = []
    mlp.train_classification(X_train, y_train, X_validation, y_validation)
    for x in X_test:
        x = np.hstack([[-1], x])
        predictions_mlp.append(np.argmax(mlp.predict_classification(x)))
    bp = 1

# Classificação com múltiplas classes