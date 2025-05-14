# Imports
import numpy as np
import matplotlib.pyplot as plt
from modelos_RNA.simple_perceptron import SimplePerceptron
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

accuracies_simple_perceptron = []
simple_perceptron_instances = []
accuracies_adaline = []
adaline_instances = []
accuracies_mlp = []
mlp_instances = []
sensitivities_adaline = []
sensitivities_mlp = []
specificities_adaline = []
specificities_mlp = []

for round in range(Config.CLASSIFICATION_N_ROUNDS):
    # Instanciando os modelos
    adaline = Adaline(η=Config.CLASSIFICATION_LEARNING_RATE, ϵ=Config.CLASSIFICATION_EPSILON, epochs=Config.CLASSIFICATION_EPOCHS)
    mlp = MLP(input_size=3, q=[4, 2], m=1, η=Config.CLASSIFICATION_LEARNING_RATE, ϵ=Config.CLASSIFICATION_EPSILON, tolleration=10, epochs=Config.CLASSIFICATION_EPOCHS)
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
    # Perceptron Simples
    perceptron = SimplePerceptron(η=Config.CLASSIFICATION_LEARNING_RATE, epochs=Config.CLASSIFICATION_EPOCHS)
    perceptron.fit(X_train, y_train)
    predictions_perceptron = perceptron.predict(X_test)
    # Adaline
    adaline.fit(X_train, y_train)
    predictions_adaline = adaline.predict_classification(X_test)
    # MLP
    # Codificando as classes
    # y_train = [np.array([1, 0]) if i == 1 else np.array([0, 1]) for i in y_train]
    # y_test = [np.array([1, 0]) if i == 1 else np.array([0, 1]) for i in y_test]
    # y_validation = [np.array([1, 0]) if i == 1 else np.array([0, 1]) for i in y_validation]
    predictions_mlp = []
    mlp.train_classification(X_train, y_train, X_validation, y_validation)
    for x in X_test:
        x = np.hstack([[-1], x])
        prediction = mlp.predict_classification(x)
        # if prediction[0] > prediction[1]:
        #     predictions_mlp.append(1)
        # else:
        #     predictions_mlp.append(0)
        predictions_mlp.append(1 if prediction > 0.5 else 0)
        # predictions_mlp.append(1 if mlp.predict_classification(x) > 0.5 else 0)
    # Calculo da acurácia
    # Acurácia
    correct_predictions_perceptron = 0
    for i in range(len(predictions_perceptron)):
        if predictions_perceptron[i] == y_test[i]:
            correct_predictions_perceptron += 1
    accuracy_perceptron = correct_predictions_perceptron / len(predictions_perceptron)
    accuracies_simple_perceptron.append(accuracy_perceptron)
    simple_perceptron_instances.append(perceptron)

    correct_predictions_adaline = 0
    for i in range(len(predictions_adaline)):
        if predictions_adaline[i] == y_test[i]:
            correct_predictions_adaline += 1
    accuracy_adaline = correct_predictions_adaline / len(predictions_adaline)
    accuracies_adaline.append(accuracy_adaline)
    adaline_instances.append(adaline)

    correct_predictions_mlp = 0
    for i in range(len(predictions_mlp)):
        if predictions_mlp[i] == y_test[i]:
            correct_predictions_mlp += 1
    accuracy_mlp = correct_predictions_mlp / len(predictions_mlp)
    accuracies_mlp.append(accuracy_mlp)
    mlp_instances.append(mlp)

# Plotando os gráficos
if Config.CLASSIFICATION_PLOT_GRAPH:
    max_simple_perceptron = simple_perceptron_instances[np.max(accuracies_simple_perceptron)]
    min_simple_perceptron = simple_perceptron_instances[np.min(accuracies_simple_perceptron)]
    max_adaline = adaline_instances[np.max(accuracies_adaline)]
    min_adaline = adaline_instances[np.min(accuracies_adaline)]
    max_mlp = mlp_instances[np.max(accuracies_mlp)]
    min_mlp = mlp_instances[np.min(accuracies_mlp)]

    # plt.figure(1)
    # plt.title('Curva de Aprendizado - Perceptron Simples (Maior Acurácia)')
    # plt.plot(max_simple_perceptron.get_epoch(), max_simple_perceptron.get_accuracy(), label='Acurácia')
    # plt.figure(2)
    # plt.title('Curva de Aprendizado - Perceptron Simples (Menor Acurácia)')
    # plt.plot(min_simple_perceptron.get_epoch(), min_simple_perceptron.get_accuracy(), label='Acurácia')
    plt.figure(3)
    plt.title('Curva de Aprendizado - Adaline (Maior Acurácia)')
    max_adaline.plot_EQMs()
    plt.figure(4)
    plt.title('Curva de Aprendizado - Adaline (Menor Acurácia)')
    min_adaline.plot_EQMs()
    plt.figure(5)
    plt.title('Curva de Aprendizado - MLP (Maior Acurácia)')
    max_mlp.plot_EQMs()
    plt.figure(6)
    plt.title('Curva de Aprendizado - MLP (Menor Acurácia)')
    min_mlp.plot_EQMs()
    plt.show()