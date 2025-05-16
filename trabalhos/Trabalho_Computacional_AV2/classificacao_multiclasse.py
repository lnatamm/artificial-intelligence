# Imports
import numpy as np
import matplotlib.pyplot as plt
from modelos_RNA.adaline import Adaline
from modelos_RNA.multi_layer_perceptron import MLP
from config import Config
import csv
import seaborn as sns

nfig = 0

coluna_vertebral = None
with open("trabalhos\\Trabalho_Computacional_AV2\\datasets\\coluna_vertebral.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    coluna_vertebral = np.array(list(reader))
# Classificação com múltiplas classes
#coluna_vertebral = np.genfromtxt("trabalhos\\Trabalho_Computacional_AV2\\datasets\\coluna_vertebral.csv", delimiter=",", dtype=None, encoding='utf-8')
# Separação das variável dependente e independentes
X = coluna_vertebral[:, :-1]  # Variável independente (todas as colunas menos a última)
# Converte X para float
X = X.astype(float)

y = coluna_vertebral[:, -1]   # Variável dependente (última coluna)
y = y.reshape(-1, 1) # Reshape para 1 coluna

y = np.where(y == 'NO', [+1, 0, 0], y) # Normal
y = np.where(y == 'DH', [0, +1, 0], y) # Disk hernia
y = np.where(y == 'SL', [0, 0, +1], y) # Spondylolisthesis

# Converte y para float
y = y.astype(float)

# Normalização dos dados
X = (X - np.min(X)) / (np.max(X) - np.min(X))
# X = 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1

mlp_instances = []
adaline_instances = []

confusion_matrices_mlp = []
confusion_matrices_adaline = []
accuracies_mlp = []
accuracies_adaline = []

for round in range(Config.CLASSIFICATION_N_ROUNDS):
    # Instanciando os modelos
    adaline = Adaline(η=Config.CLASSIFICATION_LEARNING_RATE, input_size=6, m=3, ϵ=Config.CLASSIFICATION_EPSILON, epochs=Config.CLASSIFICATION_EPOCHS)
    mlp = MLP(input_size=6, q=[2], m=3, η=Config.CLASSIFICATION_LEARNING_RATE, ϵ=Config.CLASSIFICATION_EPSILON, tolleration=10, epochs=Config.CLASSIFICATION_EPOCHS)
    mlp_underfitted = MLP(input_size=6, q=[1], m=3, η=Config.CLASSIFICATION_LEARNING_RATE, ϵ=Config.CLASSIFICATION_EPSILON, tolleration=10, epochs=Config.CLASSIFICATION_EPOCHS)
    mlp_overfitted = MLP(input_size=6, q=[32, 16, 8], m=3, η=Config.CLASSIFICATION_LEARNING_RATE, ϵ=Config.CLASSIFICATION_EPSILON, tolleration=10, epochs=Config.CLASSIFICATION_EPOCHS)
    print(f"Rodada: {round + 1}")
    # Aleatorização dos dados
    index = np.random.permutation(coluna_vertebral.shape[0])
    X_shuffled = X[index]
    y_shuffled = y[index]
    # Separação dos dados em treino, teste e validação
    train_size = int(Config.CLASSIFICATION_TRAIN * len(X))
    validation_size = int(Config.CLASSIFICATION_VALIDATION * len(X))
    # 80% para treino
    X_train = X_shuffled[:train_size]
    y_train = y_shuffled[:train_size]
    # Calcula quanto de cada classe tem no conjunto de treino
    print("Classe 0: ", np.sum(y_train[:, 0]))
    print("Classe 1: ", np.sum(y_train[:, 1]))
    print("Classe 2: ", np.sum(y_train[:, 2]))
    # 20% para teste
    X_test = X_shuffled[train_size:]
    y_test = y_shuffled[train_size:]
    # 10% para validação
    X_validation = X_test[:validation_size]
    y_validation = y_test[:validation_size]
    # Etapa de treinamento
    # adaline.fit(X_train, y_train)
    # mlp.train(X_train, y_train, X_validation, y_validation)
    adaline.fit_multiclass(X_train, y_train)
    adaline_predictions = adaline.predict_multiclass(X_test)
    adaline_predictions = np.argmax(adaline_predictions, axis=1)
    
    mlp.train(X_train, y_train, X_validation, y_validation)
    mlp_instances.append(mlp)
    adaline_instances.append(adaline)
    # Etapa de teste
    mlp_predictions = []
    for x in X_test:
        x = np.hstack([[-1], x])
        prediction = mlp.predict_classification(x.reshape(-1, 1))
        # if prediction[0] > prediction[1]:
        #     predictions_mlp.append(1)
        # else:
        #     predictions_mlp.append(0)
        mlp_predictions.append(np.argmax(prediction))
    if round == 0:
        mlp_overfitted.train(X_train, y_train, X_validation, y_validation)
        mlp_underfitted.train(X_train, y_train, X_validation, y_validation)
        predictions_mlp_overfitted = []
        predictions_mlp_underfitted = []
        for x in X_test:
            x = np.hstack([[-1], x])
            prediction_overfitted = mlp_overfitted.predict_regression(x.reshape(-1, 1))
            prediction_underfitted = mlp_underfitted.predict_regression(x.reshape(-1, 1))
            predictions_mlp_overfitted.append(np.argmax(prediction_overfitted))
            predictions_mlp_underfitted.append(np.argmax(prediction_underfitted))
        
        mlp_overfitted_class_0_0 = 0
        mlp_overfitted_class_0_1 = 0
        mlp_overfitted_class_0_2 = 0
        mlp_overfitted_class_1_0 = 0
        mlp_overfitted_class_1_1 = 0
        mlp_overfitted_class_1_2 = 0
        mlp_overfitted_class_2_0 = 0
        mlp_overfitted_class_2_1 = 0
        mlp_overfitted_class_2_2 = 0
        for i in range(len(predictions_mlp_overfitted)):
            if predictions_mlp_overfitted[i] == 0 and np.argmax(y_test[i]) == 0:
                mlp_overfitted_class_0_0 += 1
            elif predictions_mlp_overfitted[i] == 0 and np.argmax(y_test[i]) == 1:
                mlp_overfitted_class_0_1 += 1
            elif predictions_mlp_overfitted[i] == 0 and np.argmax(y_test[i]) == 2:
                mlp_overfitted_class_0_2 += 1
            elif predictions_mlp_overfitted[i] == 1 and np.argmax(y_test[i]) == 0:
                mlp_overfitted_class_1_0 += 1
            elif predictions_mlp_overfitted[i] == 1 and np.argmax(y_test[i]) == 1:
                mlp_overfitted_class_1_1 += 1
            elif predictions_mlp_overfitted[i] == 1 and np.argmax(y_test[i]) == 2:
                mlp_overfitted_class_1_2 += 1
            elif predictions_mlp_overfitted[i] == 2 and np.argmax(y_test[i]) == 0:
                mlp_overfitted_class_2_0 += 1
            elif predictions_mlp_overfitted[i] == 2 and np.argmax(y_test[i]) == 1:
                mlp_overfitted_class_2_1 += 1
            elif predictions_mlp_overfitted[i] == 2 and np.argmax(y_test[i]) == 2:
                mlp_overfitted_class_2_2 += 1
        # Matriz de confusão
        confusion_matrix_mlp_overfitted = [
            [mlp_overfitted_class_0_0, mlp_overfitted_class_0_1, mlp_overfitted_class_0_2],
            [mlp_overfitted_class_1_0, mlp_overfitted_class_1_1, mlp_overfitted_class_1_2],
            [mlp_overfitted_class_2_0, mlp_overfitted_class_2_1, mlp_overfitted_class_2_2],
        ]

        mlp_underfitted_class_0_0 = 0
        mlp_underfitted_class_0_1 = 0
        mlp_underfitted_class_0_2 = 0
        mlp_underfitted_class_1_0 = 0
        mlp_underfitted_class_1_1 = 0
        mlp_underfitted_class_1_2 = 0
        mlp_underfitted_class_2_0 = 0
        mlp_underfitted_class_2_1 = 0
        mlp_underfitted_class_2_2 = 0
        for i in range(len(predictions_mlp_overfitted)):
            if predictions_mlp_underfitted[i] == 0 and np.argmax(y_test[i]) == 0:
                mlp_underfitted_class_0_0 += 1
            elif predictions_mlp_underfitted[i] == 0 and np.argmax(y_test[i]) == 1:
                mlp_underfitted_class_0_1 += 1
            elif predictions_mlp_underfitted[i] == 0 and np.argmax(y_test[i]) == 2:
                mlp_underfitted_class_0_2 += 1
            elif predictions_mlp_underfitted[i] == 1 and np.argmax(y_test[i]) == 0:
                mlp_underfitted_class_1_0 += 1
            elif predictions_mlp_underfitted[i] == 1 and np.argmax(y_test[i]) == 1:
                mlp_underfitted_class_1_1 += 1
            elif predictions_mlp_underfitted[i] == 1 and np.argmax(y_test[i]) == 2:
                mlp_underfitted_class_1_2 += 1
            elif predictions_mlp_underfitted[i] == 2 and np.argmax(y_test[i]) == 0:
                mlp_underfitted_class_2_0 += 1
            elif predictions_mlp_underfitted[i] == 2 and np.argmax(y_test[i]) == 1:
                mlp_underfitted_class_2_1 += 1
            elif predictions_mlp_underfitted[i] == 2 and np.argmax(y_test[i]) == 2:
                mlp_underfitted_class_2_2 += 1
        # Matriz de confusão
        confusion_matrix_mlp_underfitted = [
            [mlp_underfitted_class_0_0, mlp_underfitted_class_0_1, mlp_underfitted_class_0_2],
            [mlp_underfitted_class_1_0, mlp_underfitted_class_1_1, mlp_underfitted_class_1_2],
            [mlp_underfitted_class_2_0, mlp_underfitted_class_2_1, mlp_underfitted_class_2_2],
        ]
        plt.figure(nfig)
        nfig += 1
        plt.title('MLP Superfitted')
        mlp_overfitted.plot_EQMs("Superfitted")
        plt.figure(nfig)
        nfig += 1
        plt.title('MLP Underfitted')
        mlp_underfitted.plot_EQMs("Underfitted")
        plt.figure(nfig)
        nfig += 1

        accuracy_mlp_overfitted = (mlp_overfitted_class_0_0 + mlp_overfitted_class_1_1 + mlp_overfitted_class_2_2) / len(y_test)
        accuracy_mlp_underfitted = (mlp_underfitted_class_0_0 + mlp_underfitted_class_1_1 + mlp_underfitted_class_2_2) / len(y_test)

        sns.heatmap(confusion_matrix_mlp_overfitted, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'Matriz de Confusão MLP Superfitted - Acurácia: {accuracy_mlp_overfitted:.2f}')
        plt.figure(nfig)
        nfig += 1

        sns.heatmap(confusion_matrix_mlp_underfitted, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'Matriz de Confusão MLP Underfitted - Acurácia: {accuracy_mlp_underfitted:.2f}')

        plt.show()
    # for prediction in adaline_predictions:
    #     # if prediction[0] > prediction[1]:
    #     #     predictions_adaline.append(1)
    #     # else:
    #     #     predictions_adaline.append(0)
    #     adaline_predictions.append(np.argmax(prediction))

    mlp_class_0_0 = 0
    mlp_class_0_1 = 0
    mlp_class_0_2 = 0
    mlp_class_1_0 = 0
    mlp_class_1_1 = 0
    mlp_class_1_2 = 0
    mlp_class_2_0 = 0
    mlp_class_2_1 = 0
    mlp_class_2_2 = 0

    for i in range(len(mlp_predictions)):
        if mlp_predictions[i] == 0 and np.argmax(y_test[i]) == 0:
            mlp_class_0_0 += 1
        elif mlp_predictions[i] == 0 and np.argmax(y_test[i]) == 1:
            mlp_class_0_1 += 1
        elif mlp_predictions[i] == 0 and np.argmax(y_test[i]) == 2:
            mlp_class_0_2 += 1
        elif mlp_predictions[i] == 1 and np.argmax(y_test[i]) == 0:
            mlp_class_1_0 += 1
        elif mlp_predictions[i] == 1 and np.argmax(y_test[i]) == 1:
            mlp_class_1_1 += 1
        elif mlp_predictions[i] == 1 and np.argmax(y_test[i]) == 2:
            mlp_class_1_2 += 1
        elif mlp_predictions[i] == 2 and np.argmax(y_test[i]) == 0:
            mlp_class_2_0 += 1
        elif mlp_predictions[i] == 2 and np.argmax(y_test[i]) == 1:
            mlp_class_2_1 += 1
        elif mlp_predictions[i] == 2 and np.argmax(y_test[i]) == 2:
            mlp_class_2_2 += 1
    # Matriz de confusão
    confusion_matrix_mlp = [
        [mlp_class_0_0, mlp_class_0_1, mlp_class_0_2],
        [mlp_class_1_0, mlp_class_1_1, mlp_class_1_2],
        [mlp_class_2_0, mlp_class_2_1, mlp_class_2_2],
    ]

    accuracy_mlp = (mlp_class_0_0 + mlp_class_1_1 + mlp_class_2_2) / len(y_test)
    accuracies_mlp.append(accuracy_mlp)

    confusion_matrices_mlp.append(confusion_matrix_mlp)

    # for i in range(len(adaline_predictions)):
    #     if adaline_predictions[i] == 0 and y_test[i] == 0:
    #         class_0_0 += 1
    #     elif adaline_predictions[i] == 0 and y_test[i] == 1:
    #         class_0_1 += 1
    #     elif adaline_predictions[i] == 0 and y_test[i] == 2:
    #         class_0_2 += 1
    #     elif adaline_predictions[i] == 1 and y_test[i] == 0:
    #         class_1_0 += 1
    #     elif adaline_predictions[i] == 1 and y_test[i] == 1:
    #         class_1_1 += 1
    #     elif adaline_predictions[i] == 1 and y_test[i] == 2:
    #         class_1_2 += 1
    #     elif adaline_predictions[i] == 2 and y_test[i] == 0:
    #         class_2_0 += 1
    #     elif adaline_predictions[i] == 2 and y_test[i] == 1:
    #         class_2_1 += 1
    #     elif adaline_predictions[i] == 2 and y_test[i] == 2:
    #         class_2_2 += 1
    
    adaline_class_0_0 = 0
    adaline_class_0_1 = 0
    adaline_class_0_2 = 0
    adaline_class_1_0 = 0
    adaline_class_1_1 = 0
    adaline_class_1_2 = 0
    adaline_class_2_0 = 0
    adaline_class_2_1 = 0
    adaline_class_2_2 = 0

    for i in range(len(mlp_predictions)):
        if adaline_predictions[i] == 0 and np.argmax(y_test[i]) == 0:
            adaline_class_0_0 += 1
        elif adaline_predictions[i] == 0 and np.argmax(y_test[i]) == 1:
            adaline_class_0_1 += 1
        elif adaline_predictions[i] == 0 and np.argmax(y_test[i]) == 2:
            adaline_class_0_2 += 1
        elif adaline_predictions[i] == 1 and np.argmax(y_test[i]) == 0:
            adaline_class_1_0 += 1
        elif adaline_predictions[i] == 1 and np.argmax(y_test[i]) == 1:
            adaline_class_1_1 += 1
        elif adaline_predictions[i] == 1 and np.argmax(y_test[i]) == 2:
            adaline_class_1_2 += 1
        elif adaline_predictions[i] == 2 and np.argmax(y_test[i]) == 0:
            adaline_class_2_0 += 1
        elif adaline_predictions[i] == 2 and np.argmax(y_test[i]) == 1:
            adaline_class_2_1 += 1
        elif adaline_predictions[i] == 2 and np.argmax(y_test[i]) == 2:
            adaline_class_2_2 += 1

    # Matriz de confusão
    confusion_matrix_adaline = [
        [adaline_class_0_0, adaline_class_0_1, adaline_class_0_2],
        [adaline_class_1_0, adaline_class_1_1, adaline_class_1_2],
        [adaline_class_2_0, adaline_class_2_1, adaline_class_2_2],
    ]

    accuracy_adaline = (adaline_class_0_0 + adaline_class_1_1 + adaline_class_2_2) / len(y_test)
    accuracies_adaline.append(accuracy_adaline)

    confusion_matrices_adaline.append(confusion_matrix_adaline)

# Plotando os gráficos
if Config.CLASSIFICATION_PLOT_GRAPH:

    max_mlp_index = np.argmax(accuracies_mlp)
    min_mlp_index = np.argmin(accuracies_mlp)
    max_mlp = mlp_instances[max_mlp_index]
    min_mlp = mlp_instances[min_mlp_index]
    
    max_adaline_index = np.argmax(accuracies_adaline)
    min_adaline_index = np.argmin(accuracies_adaline)
    max_adaline = adaline_instances[max_adaline_index]
    min_adaline = adaline_instances[min_adaline_index]

    max_accuracy_mlp = accuracies_mlp[max_mlp_index]
    min_accuracy_mlp = accuracies_mlp[min_mlp_index]

    max_accuracy_adaline = accuracies_adaline[max_adaline_index]
    min_accuracy_adaline = accuracies_adaline[min_adaline_index]

    max_confusion_matrix_mlp = confusion_matrices_mlp[max_mlp_index]
    min_confusion_matrix_mlp = confusion_matrices_mlp[min_mlp_index]

    max_confusion_matrix_adaline = confusion_matrices_adaline[max_adaline_index]
    min_confusion_matrix_adaline = confusion_matrices_adaline[min_adaline_index]
    
    plt.figure(nfig)
    nfig += 1
    max_mlp.plot_EQMs("Mlp - Máxima Acurácia")
    plt.figure(nfig)
    nfig += 1
    min_mlp.plot_EQMs("Mlp - Mínima Acurácia")
    plt.figure(nfig)
    nfig += 1
    max_adaline.plot_EQMs("Adaline - Máxima Acurácia")
    plt.figure(nfig)
    nfig += 1
    min_adaline.plot_EQMs("Adaline - Mínima Acurácia")
    plt.figure(nfig)

    metrics_accuracy_adaline = {
        "mean_accuracy": np.mean(accuracies_adaline),
        "std_accuracy": np.std(accuracies_adaline),
        "max_accuracy": np.max(accuracies_adaline),
        "min_accuracy": np.min(accuracies_adaline),
    }

    metrics_accuracy_mlp = {
        "mean_accuracy": np.mean(accuracies_mlp),
        "std_accuracy": np.std(accuracies_mlp),
        "max_accuracy": np.max(accuracies_mlp),
        "min_accuracy": np.min(accuracies_mlp),
    }

    models = [
        "Adaline",
        "MLP",
    ]

    metrics_accuracy = {
        "Média Acurácia": 
            (
                metrics_accuracy_adaline["mean_accuracy"],
                metrics_accuracy_mlp["mean_accuracy"],
            ),
        "Desvio Padrão Acurácia": 
            (
                metrics_accuracy_adaline["std_accuracy"],
                metrics_accuracy_mlp["std_accuracy"],
            ),
        "Máximo Acurácia": 
            (
                metrics_accuracy_adaline["max_accuracy"],
                metrics_accuracy_mlp["max_accuracy"],
            ),
        "Mínimo Acurácia": 
            (
                metrics_accuracy_adaline["min_accuracy"],
                metrics_accuracy_mlp["min_accuracy"],
            ),
    }


    x = np.arange(len(models))  # the label locations
    largura = 0.2  # the width of the bars
    mult = 0
    
    fig, ax = plt.subplots(layout='constrained')
    nfig += 1

    for tipo, medida in metrics_accuracy.items():
        offset = largura * mult
        rects = ax.bar(x + offset, medida, largura, label=tipo)
        ax.bar_label(rects, padding=3)
        mult += 1

    ax.set_title('Métricas de Acurácia dos Modelos')
    ax.set_xticks(x + largura, models)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.1)
    plt.show()
    
    sns.heatmap(max_confusion_matrix_mlp, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f'Matriz de Confusão - MLP (Máxima Acurácia) - Acurácia: {max_accuracy_mlp:.2f}')
    nfig += 100
    plt.figure(nfig)
    nfig += 1
    sns.heatmap(min_confusion_matrix_mlp, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f'Matriz de Confusão - MLP (Mínima Acurácia) - Acurácia: {min_accuracy_mlp:.2f}')
    plt.figure(nfig)
    nfig += 1
    sns.heatmap(max_confusion_matrix_adaline, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f'Matriz de Confusão - Adaline (Máxima Acurácia) - Acurácia: {max_accuracy_adaline:.2f}')
    plt.figure(nfig)
    nfig += 1
    sns.heatmap(min_confusion_matrix_adaline, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f'Matriz de Confusão - Adaline (Mínima Acurácia) - Acurácia: {min_accuracy_adaline:.2f}')

    plt.show()
    # plt.figure(nfig)
    # nfig += 1
    # sns.heatmap(min_confusion_matrix_simple_perceptron, annot=True, fmt=".2f", cmap="Blues")
    # plt.title(f'Matriz de Confusão - Perceptron Simples (Menor Acurácia) - Acurácia: {min_accuracy_simple_perceptron:.2f} - Sensibilidade: {min_sensitivity_simple_perceptron:.2f} - Especificidade: {min_specificity_simple_perceptron:.2f}')
    # plt.figure(nfig)
    # nfig += 1
    # sns.heatmap(max_confusion_matrix_simple_perceptron, annot=True, fmt=".2f", cmap="Blues")
    # plt.title(f'Matriz de Confusão - Perceptron Simples (Maior Acurácia) - Acurácia: {max_accuracy_simple_perceptron:.2f} - Sensibilidade: {max_sensitivity_simple_perceptron:.2f} - Especificidade: {max_specificity_simple_perceptron:.2f}')
    # plt.figure(nfig)
    # nfig += 1
    # sns.heatmap(min_confusion_matrix_adaline, annot=True, fmt=".2f", cmap="Blues")
    # plt.title(f'Matriz de Confusão - Adaline (Menor Acurácia) - Acurácia: {min_accuracy_adaline:.2f} - Sensibilidade: {min_sensitivity_adaline:.2f} - Especificidade: {min_specificity_adaline:.2f}')
    # plt.figure(nfig)
    # nfig += 1
    # sns.heatmap(max_confusion_matrix_adaline, annot=True, fmt=".2f", cmap="Blues")
    # plt.title(f'Matriz de Confusão - Adaline (Maior Acurácia) - Acurácia: {max_accuracy_adaline:.2f} - Sensibilidade: {max_sensitivity_adaline:.2f} - Especificidade: {max_specificity_adaline:.2f}')
    
    # plt.figure(nfig)
    # nfig += 1
    # sns.heatmap(min_confusion_matrix_mlp, annot=True, fmt=".2f", cmap="Blues")
    # plt.title(f'Matriz de Confusão - MLP (Menor Acurácia) - Acurácia: {min_accuracy_mlp:.2f} - Sensibilidade: {min_sensitivity_mlp:.2f} - Especificidade: {min_specificity_mlp:.2f}')
    # plt.figure(nfig)
    # nfig += 1
    # sns.heatmap(max_confusion_matrix_mlp, annot=True, fmt=".2f", cmap="Blues")
    # plt.title(f'Matriz de Confusão - MLP (Maior Acurácia) - Acurácia: {max_accuracy_mlp:.2f} - Sensibilidade: {max_sensitivity_mlp:.2f} - Especificidade: {max_specificity_mlp:.2f}')

    # plt.figure(nfig)
    # nfig += 1
    # max_simple_perceptron.plot_errors('Curva de Aprendizado - Perceptron Simples (Maior Acurácia)')
    # plt.figure(nfig)
    # nfig += 1
    # min_simple_perceptron.plot_errors('Curva de Aprendizado - Perceptron Simples (Menor Acurácia)')
    # plt.figure(nfig)
    # nfig += 1
    # max_adaline.plot_EQMs('Curva de Aprendizado - Adaline (Maior Acurácia)')
    # plt.figure(nfig)
    # nfig += 1
    # min_adaline.plot_EQMs('Curva de Aprendizado - Adaline (Menor Acurácia)')
    # plt.figure(nfig)
    # nfig += 1
    # max_mlp.plot_EQMs('Curva de Aprendizado - MLP (Maior Acurácia)')
    # plt.figure(nfig)
    # nfig += 1
    # min_mlp.plot_EQMs('Curva de Aprendizado - MLP (Menor Acurácia)')
    # plt.show()


# if Config.CLASSIFICATION_PLOT_GRAPH:
#     # Faz o plot 3d dos dados
#     fig = plt.figure(0)
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_title('Dados Originais')
#     ax.set_xlabel('X1')
#     ax.set_ylabel('X2')
#     ax.set_zlabel('X3')
#     ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', label='Dados Originais')