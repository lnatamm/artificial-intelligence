# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modelos_RNA.simple_perceptron import SimplePerceptron
from modelos_RNA.adaline import Adaline
from modelos_RNA.multi_layer_perceptron import MLP
from config import Config

# Leitura dos dados
spiral = np.loadtxt("trabalhos\\Trabalho_Computacional_AV2\\datasets\\Spiral3d.csv", delimiter=",")

nfig = 0

# Classificação Linear Simples
# Separação das variável dependente e independentes
X = spiral[:, :-1]  # Variável independente (todas as colunas menos a última)
y = spiral[:, -1]   # Variável dependente (última coluna)

if Config.CLASSIFICATION_PLOT_GRAPH:
    # Faz o plot 3d dos dados
    fig = plt.figure(nfig)
    nfig += 1
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
    fig = plt.figure(nfig)
    nfig += 1
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Dados Normalizados')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', label='Dados Normalizados')

plt.show()

accuracies_simple_perceptron = []
sensitivities_simple_perceptron = []
specificities_simple_perceptron = []
confusion_matrices_simple_perceptron = []
simple_perceptron_instances = []
accuracies_adaline = []
sensitivities_adaline = []
specificities_adaline = []
confusion_matrices_adaline = []
adaline_instances = []
accuracies_mlp = []
sensitivities_mlp = []
specificities_mlp = []
confusion_matrices_mlp = []
mlp_instances = []

for round in range(Config.CLASSIFICATION_N_ROUNDS):
    # Instanciando os modelos
    adaline = Adaline(η=Config.CLASSIFICATION_LEARNING_RATE, ϵ=Config.CLASSIFICATION_EPSILON, epochs=200)
    mlp = MLP(
        input_size=3,
        q=Config.CLASSIFICATION_MLP_LAYERS,
        m=Config.CLASSIFICATION_MLP_OUTPUT_SIZE,
        η=Config.CLASSIFICATION_LEARNING_RATE,
        ϵ=Config.CLASSIFICATION_EPSILON,
        tolleration=Config.CLASSIFICATION_MLP_TOLLERATION,
        epochs=Config.CLASSIFICATION_EPOCHS
    )
    mlp_overfitted = MLP(
        input_size=3,
        q=Config.CLASSIFICATION_MLP_OVERFITTED_LAYERS,
        m=Config.CLASSIFICATION_MLP_OUTPUT_SIZE,
        η=Config.CLASSIFICATION_LEARNING_RATE,
        ϵ=Config.CLASSIFICATION_EPSILON,
        tolleration=Config.CLASSIFICATION_MLP_TOLLERATION,
        epochs=Config.CLASSIFICATION_EPOCHS
    )
    mlp_underfitted = MLP(
        input_size=3,
        q=Config.CLASSIFICATION_MLP_UNDERFITTED_LAYERS,
        m=Config.CLASSIFICATION_MLP_OUTPUT_SIZE,
        η=Config.CLASSIFICATION_LEARNING_RATE,
        ϵ=Config.CLASSIFICATION_EPSILON,
        tolleration=Config.CLASSIFICATION_MLP_TOLLERATION,
        epochs=Config.CLASSIFICATION_EPOCHS
    )
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
    perceptron = SimplePerceptron(η=Config.CLASSIFICATION_LEARNING_RATE, epochs=200)
    perceptron.fit(X_train, y_train)
    predictions_perceptron = perceptron.predict(X_test)
    predictions_perceptron = np.where(predictions_perceptron == -1, 0, predictions_perceptron)
    # Adaline
    adaline.fit(X_train, y_train)
    predictions_adaline = adaline.predict_classification(X_test)
    predictions_adaline = np.where(predictions_adaline == -1, 0, predictions_adaline)
    predictions_mlp = []
    mlp.train(X_train, y_train, X_validation, y_validation)
    for x in X_test:
        x = np.hstack([[-1], x])
        prediction = mlp.predict_regression(x.reshape(-1, 1))
        predictions_mlp.append(1 if prediction.item() >= 0.5 else 0)
    if round == 0:
        mlp_overfitted.train(X_train, y_train, X_validation, y_validation)
        mlp_underfitted.train(X_train, y_train, X_validation, y_validation)
        predictions_mlp_overfitted = []
        predictions_mlp_underfitted = []
        for x in X_test:
            x = np.hstack([[-1], x])
            prediction_overfitted = mlp_overfitted.predict_regression(x.reshape(-1, 1))
            prediction_underfitted = mlp_underfitted.predict_regression(x.reshape(-1, 1))
            predictions_mlp_overfitted.append(1 if prediction_overfitted.item() >= 0.5 else 0)
            predictions_mlp_underfitted.append(1 if prediction_underfitted.item() >= 0.5 else 0)
        confusion_matrix_mlp_overfitted = np.zeros((2, 2))
        true_positive_mlp_overfitted = 0
        true_negative_mlp_overfitted = 0
        false_positive_mlp_overfitted = 0
        false_negative_mlp_overfitted = 0
        for i in range(len(predictions_mlp_overfitted)):
            if predictions_mlp_overfitted[i] == 1 and y_test[i] == 1:
                true_positive_mlp_overfitted += 1
            elif predictions_mlp_overfitted[i] == 0 and y_test[i] == 0:
                true_negative_mlp_overfitted += 1
            elif predictions_mlp_overfitted[i] == 1 and y_test[i] == 0:
                false_positive_mlp_overfitted += 1
            elif predictions_mlp_overfitted[i] == 0 and y_test[i] == 1:
                false_negative_mlp_overfitted += 1
        confusion_matrix_mlp_overfitted[0][0] = true_positive_mlp_overfitted
        confusion_matrix_mlp_overfitted[0][1] = false_positive_mlp_overfitted
        confusion_matrix_mlp_overfitted[1][0] = false_negative_mlp_overfitted
        confusion_matrix_mlp_overfitted[1][1] = true_negative_mlp_overfitted

        confusion_matrix_mlp_underfitted = np.zeros((2, 2))
        true_positive_mlp_underfitted = 0
        true_negative_mlp_underfitted = 0
        false_positive_mlp_underfitted = 0
        false_negative_mlp_underfitted = 0
        for i in range(len(predictions_mlp_underfitted)):
            if predictions_mlp_underfitted[i] == 1 and y_test[i] == 1:
                true_positive_mlp_underfitted += 1
            elif predictions_mlp_underfitted[i] == 0 and y_test[i] == 0:
                true_negative_mlp_underfitted += 1
            elif predictions_mlp_underfitted[i] == 1 and y_test[i] == 0:
                false_positive_mlp_underfitted += 1
            elif predictions_mlp_underfitted[i] == 0 and y_test[i] == 1:
                false_negative_mlp_underfitted += 1
        confusion_matrix_mlp_underfitted[0][0] = true_positive_mlp_underfitted
        confusion_matrix_mlp_underfitted[0][1] = false_positive_mlp_underfitted
        confusion_matrix_mlp_underfitted[1][0] = false_negative_mlp_underfitted
        confusion_matrix_mlp_underfitted[1][1] = true_negative_mlp_underfitted
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

        accuracy_mlp_overfitted = (true_positive_mlp_overfitted + true_negative_mlp_overfitted) / (true_positive_mlp_overfitted + false_positive_mlp_overfitted + true_negative_mlp_overfitted + false_negative_mlp_overfitted)
        accuracy_mlp_underfitted = (true_positive_mlp_underfitted + true_negative_mlp_underfitted) / (true_positive_mlp_underfitted + false_positive_mlp_underfitted + true_negative_mlp_underfitted + false_negative_mlp_underfitted)

        sensitivity_mlp_overfitted = true_positive_mlp_overfitted / (true_positive_mlp_overfitted + false_negative_mlp_overfitted)
        sensitivity_mlp_underfitted = true_positive_mlp_underfitted / (true_positive_mlp_underfitted + false_negative_mlp_underfitted)

        specificity_mlp_overfitted = true_negative_mlp_overfitted / (true_negative_mlp_overfitted + false_positive_mlp_overfitted)
        specificity_mlp_underfitted = true_negative_mlp_underfitted / (true_negative_mlp_underfitted + false_positive_mlp_underfitted)

        sns.heatmap(confusion_matrix_mlp_overfitted, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Matriz de Confusão MLP Superfitted')
        plt.figure(nfig)
        nfig += 1

        sns.heatmap(confusion_matrix_mlp_underfitted, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Matriz de Confusão MLP Underfitted')

        plt.show()


    # Matriz de confusão
    confusion_matrix_perceptron = np.zeros((2, 2))
    
    true_positive_perceptron = 0
    true_negative_perceptron = 0
    false_positive_perceptron = 0
    false_negative_perceptron = 0
    for i in range(len(predictions_perceptron)):

        if predictions_perceptron[i] == 1 and y_test[i] == 1:
            true_positive_perceptron += 1
        elif predictions_perceptron[i] == 0 and y_test[i] == 0:
            true_negative_perceptron += 1
        elif predictions_perceptron[i] == 1 and y_test[i] == 0:
            false_positive_perceptron += 1
        elif predictions_perceptron[i] == 0 and y_test[i] == 1:
            false_negative_perceptron += 1
    
    confusion_matrix_perceptron[0][0] = true_positive_perceptron
    confusion_matrix_perceptron[0][1] = false_positive_perceptron   
    confusion_matrix_perceptron[1][0] = false_negative_perceptron
    confusion_matrix_perceptron[1][1] = true_negative_perceptron
    
    confusion_matrices_simple_perceptron.append(confusion_matrix_perceptron)

    # Calculando as métricas
    accuracy_perceptron = (true_positive_perceptron + true_negative_perceptron) / (true_positive_perceptron + false_positive_perceptron + true_negative_perceptron + false_negative_perceptron)
    accuracies_simple_perceptron.append(accuracy_perceptron)
    
    sensitivity_perceptron = true_positive_perceptron / (true_positive_perceptron + false_negative_perceptron)
    sensitivities_simple_perceptron.append(sensitivity_perceptron)
    
    specificity_perceptron = true_negative_perceptron / (true_negative_perceptron + false_positive_perceptron)
    specificities_simple_perceptron.append(specificity_perceptron)
    
    simple_perceptron_instances.append(perceptron)

    confusion_matrix_adaline = np.zeros((2, 2))
    true_positive_adaline = 0
    true_negative_adaline = 0
    false_positive_adaline = 0
    false_negative_adaline = 0
    for i in range(len(predictions_adaline)):
        if predictions_adaline[i] == 1 and y_test[i] == 1:
            true_positive_adaline += 1
        elif predictions_adaline[i] == 0 and y_test[i] == 0:
            true_negative_adaline += 1
        elif predictions_adaline[i] == 1 and y_test[i] == 0:
            false_positive_adaline += 1
        elif predictions_adaline[i] == 0 and y_test[i] == 1:
            false_negative_adaline += 1

    confusion_matrix_adaline[0][0] = true_positive_adaline
    confusion_matrix_adaline[0][1] = false_positive_adaline
    confusion_matrix_adaline[1][0] = false_negative_adaline
    confusion_matrix_adaline[1][1] = true_negative_adaline

    confusion_matrices_adaline.append(confusion_matrix_adaline)

    # Calculando as métricas
    accuracy_adaline = (true_positive_adaline + true_negative_adaline) / (true_positive_adaline + false_positive_adaline + true_negative_adaline + false_negative_adaline)
    accuracies_adaline.append(accuracy_adaline)
    
    sensitivity_adaline = true_positive_adaline / (true_positive_adaline + false_negative_adaline)
    sensitivities_adaline.append(sensitivity_adaline)
    
    specificity_adaline = true_negative_adaline / (true_negative_perceptron + false_positive_adaline)
    specificities_adaline.append(specificity_adaline)
    adaline_instances.append(adaline)

    confusion_matrix_mlp = np.zeros((2, 2))
    true_positive_mlp = 0
    true_negative_mlp = 0
    false_positive_mlp = 0
    false_negative_mlp = 0
    for i in range(len(predictions_mlp)):
        if predictions_mlp[i] == 1 and y_test[i] == 1:
            true_positive_mlp += 1
        elif predictions_mlp[i] == 0 and y_test[i] == 0:
            true_negative_mlp += 1
        elif predictions_mlp[i] == 1 and y_test[i] == 0:
            false_positive_mlp += 1
        elif predictions_mlp[i] == 0 and y_test[i] == 1:
            false_negative_mlp += 1
    confusion_matrix_mlp[0][0] = true_positive_mlp
    confusion_matrix_mlp[0][1] = false_positive_mlp
    confusion_matrix_mlp[1][0] = false_negative_mlp
    confusion_matrix_mlp[1][1] = true_negative_mlp

    confusion_matrices_mlp.append(confusion_matrix_mlp)

    accuracy_mlp = (true_positive_mlp + true_negative_mlp) / (true_positive_mlp + false_positive_mlp + true_negative_mlp + false_negative_mlp)
    accuracies_mlp.append(accuracy_mlp)

    sensitivity_mlp = true_positive_mlp / (true_positive_mlp + false_negative_mlp)
    sensitivities_mlp.append(sensitivity_mlp)

    specificity_mlp = true_negative_mlp / (true_negative_mlp + false_positive_mlp)
    specificities_mlp.append(specificity_mlp)
    mlp_instances.append(mlp)

# Plotando os gráficos
if Config.CLASSIFICATION_PLOT_GRAPH:
    max_simple_perceptron_index = np.argmax(accuracies_simple_perceptron)
    min_simple_perceptron_index = np.argmin(accuracies_simple_perceptron)
    max_simple_perceptron = simple_perceptron_instances[max_simple_perceptron_index]
    min_simple_perceptron = simple_perceptron_instances[min_simple_perceptron_index]

    max_adaline_index = np.argmax(accuracies_adaline)
    min_adaline_index = np.argmin(accuracies_adaline)
    max_adaline = adaline_instances[max_adaline_index]
    min_adaline = adaline_instances[min_adaline_index]

    max_mlp_index = np.argmax(accuracies_mlp)
    min_mlp_index = np.argmin(accuracies_mlp)
    max_mlp = mlp_instances[max_mlp_index]
    min_mlp = mlp_instances[min_mlp_index]

    max_accuracy_simple_perceptron = accuracies_simple_perceptron[max_simple_perceptron_index]
    min_accuracy_simple_perceptron = accuracies_simple_perceptron[min_simple_perceptron_index]

    max_accuracy_adaline = accuracies_adaline[max_adaline_index]
    min_accuracy_adaline = accuracies_adaline[min_adaline_index]

    max_accuracy_mlp = accuracies_mlp[max_mlp_index]
    min_accuracy_mlp = accuracies_mlp[min_mlp_index]

    max_sensitivity_simple_perceptron = sensitivities_simple_perceptron[max_simple_perceptron_index]
    min_sensitivity_simple_perceptron = sensitivities_simple_perceptron[min_simple_perceptron_index]

    max_sensitivity_adaline = sensitivities_adaline[max_adaline_index]
    min_sensitivity_adaline = sensitivities_adaline[min_adaline_index]

    max_sensitivity_mlp = sensitivities_mlp[max_mlp_index]
    min_sensitivity_mlp = sensitivities_mlp[min_mlp_index]

    max_specificity_simple_perceptron = specificities_simple_perceptron[max_simple_perceptron_index]
    min_specificity_simple_perceptron = specificities_simple_perceptron[min_simple_perceptron_index]

    max_specificity_adaline = specificities_adaline[max_adaline_index]
    min_specificity_adaline = specificities_adaline[min_adaline_index]

    max_specificity_mlp = specificities_mlp[max_mlp_index]
    min_specificity_mlp = specificities_mlp[min_mlp_index]

    max_confusion_matrix_simple_perceptron = confusion_matrices_simple_perceptron[max_simple_perceptron_index]
    min_confusion_matrix_simple_perceptron = confusion_matrices_simple_perceptron[min_simple_perceptron_index]

    max_confusion_matrix_adaline = confusion_matrices_adaline[max_adaline_index]
    min_confusion_matrix_adaline = confusion_matrices_adaline[min_adaline_index]

    max_confusion_matrix_mlp = confusion_matrices_mlp[max_mlp_index]
    min_confusion_matrix_mlp = confusion_matrices_mlp[min_mlp_index]

    metrics_accuracy_simple_perceptron = {
        "mean_accuracy": np.mean(accuracies_simple_perceptron),
        "std_accuracy": np.std(accuracies_simple_perceptron),
        "max_accuracy": np.max(accuracies_simple_perceptron),
        "min_accuracy": np.min(accuracies_simple_perceptron),
    }
    metrics_sensitivity_simple_perceptron = {
        "mean_sensitivity": np.mean(sensitivities_simple_perceptron),
        "std_sensitivity": np.std(sensitivities_simple_perceptron),
        "max_sensitivity": np.max(sensitivities_simple_perceptron),
        "min_sensitivity": np.min(sensitivities_simple_perceptron),
    }
    metrics_specificity_simple_perceptron = {
        "mean_specificity": np.mean(specificities_simple_perceptron),
        "std_specificity": np.std(specificities_simple_perceptron),
        "max_specificity": np.max(specificities_simple_perceptron),
        "min_specificity": np.min(specificities_simple_perceptron),
    }

    metrics_accuracy_adaline = {
        "mean_accuracy": np.mean(accuracies_adaline),
        "std_accuracy": np.std(accuracies_adaline),
        "max_accuracy": np.max(accuracies_adaline),
        "min_accuracy": np.min(accuracies_adaline),
    }
    metrics_sensitivity_adaline = {
        "mean_sensitivity": np.mean(sensitivities_adaline),
        "std_sensitivity": np.std(sensitivities_adaline),
        "max_sensitivity": np.max(sensitivities_adaline),
        "min_sensitivity": np.min(sensitivities_adaline),
    }
    metrics_specificity_adaline = {
        "mean_specificity": np.mean(specificities_adaline),
        "std_specificity": np.std(specificities_adaline),
        "max_specificity": np.max(specificities_adaline),
        "min_specificity": np.min(specificities_adaline),
    }

    metrics_accuracy_mlp = {
        "mean_accuracy": np.mean(accuracies_mlp),
        "std_accuracy": np.std(accuracies_mlp),
        "max_accuracy": np.max(accuracies_mlp),
        "min_accuracy": np.min(accuracies_mlp),
    }
    metrics_sensitivity_mlp = {
        "mean_sensitivity": np.mean(sensitivities_mlp),
        "std_sensitivity": np.std(sensitivities_mlp),
        "max_sensitivity": np.max(sensitivities_mlp),
        "min_sensitivity": np.min(sensitivities_mlp),
    }
    metrics_specificity_mlp = {
        "mean_specificity": np.mean(specificities_mlp),
        "std_specificity": np.std(specificities_mlp),
        "max_specificity": np.max(specificities_mlp),
        "min_specificity": np.min(specificities_mlp),
    }

    models = [
        "Perceptron Simples",
        "Adaline",
        "MLP",
    ]

    metrics_accuracy = {
        "Média Acurácia": 
            (
                metrics_accuracy_simple_perceptron["mean_accuracy"],
                metrics_accuracy_adaline["mean_accuracy"],
                metrics_accuracy_mlp["mean_accuracy"],
            ),
        "Desvio Padrão Acurácia": 
            (
                metrics_accuracy_simple_perceptron["std_accuracy"],
                metrics_accuracy_adaline["std_accuracy"],
                metrics_accuracy_mlp["std_accuracy"],
            ),
        "Máximo Acurácia": 
            (
                metrics_accuracy_simple_perceptron["max_accuracy"],
                metrics_accuracy_adaline["max_accuracy"],
                metrics_accuracy_mlp["max_accuracy"],
            ),
        "Mínimo Acurácia": 
            (
                metrics_accuracy_simple_perceptron["min_accuracy"],
                metrics_accuracy_adaline["min_accuracy"],
                metrics_accuracy_mlp["min_accuracy"],
            ),
    }

    metrics_sensitivity = {
        "Média Sensibilidade": 
            (
                metrics_sensitivity_simple_perceptron["mean_sensitivity"],
                metrics_sensitivity_adaline["mean_sensitivity"],
                metrics_sensitivity_mlp["mean_sensitivity"],
            ),
        "Desvio Padrão Sensibilidade": 
            (
                metrics_sensitivity_simple_perceptron["std_sensitivity"],
                metrics_sensitivity_adaline["std_sensitivity"],
                metrics_sensitivity_mlp["std_sensitivity"],
            ),
        "Máximo Sensibilidade": 
            (
                metrics_sensitivity_simple_perceptron["max_sensitivity"],
                metrics_sensitivity_adaline["max_sensitivity"],
                metrics_sensitivity_mlp["max_sensitivity"],
            ),
        "Mínimo Sensibilidade": 
            (
                metrics_sensitivity_simple_perceptron["min_sensitivity"],
                metrics_sensitivity_adaline["min_sensitivity"],
                metrics_sensitivity_mlp["min_sensitivity"],
            ),
    }

    metrics_specificity = {
        "Média Especificidade": 
            (
                metrics_specificity_simple_perceptron["mean_specificity"],
                metrics_specificity_adaline["mean_specificity"],
                metrics_specificity_mlp["mean_specificity"],
            ),
        "Desvio Padrão Especificidade": 
            (
                metrics_specificity_simple_perceptron["std_specificity"],
                metrics_specificity_adaline["std_specificity"],
                metrics_specificity_mlp["std_specificity"],
            ),
        "Máximo Especificidade": 
            (
                metrics_specificity_simple_perceptron["max_specificity"],
                metrics_specificity_adaline["max_specificity"],
                metrics_specificity_mlp["max_specificity"],
            ),
        "Mínimo Especificidade": 
            (
                metrics_specificity_simple_perceptron["min_specificity"],
                metrics_specificity_adaline["min_specificity"],
                metrics_specificity_mlp["min_specificity"],
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

    mult = 0
    fig, ax = plt.subplots(layout='constrained')
    nfig += 1

    for tipo, medida in metrics_sensitivity.items():
        offset = largura * mult
        rects = ax.bar(x + offset, medida, largura, label=tipo)
        ax.bar_label(rects, padding=3)
        mult += 1

    ax.set_title('Métricas de Sensibilidade dos Modelos')
    ax.set_xticks(x + largura, models)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.1)

    mult = 0
    fig, ax = plt.subplots(layout='constrained')
    nfig += 1

    for tipo, medida in metrics_specificity.items():
        offset = largura * mult
        rects = ax.bar(x + offset, medida, largura, label=tipo)
        ax.bar_label(rects, padding=3)
        mult += 1

    ax.set_title('Métricas de Especificidade dos Modelos')
    ax.set_xticks(x + largura, models)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.1)

    plt.figure(nfig)
    nfig += 1
    sns.heatmap(min_confusion_matrix_simple_perceptron, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f'Matriz de Confusão - Perceptron Simples (Menor Acurácia) - Acurácia: {min_accuracy_simple_perceptron:.2f} - Sensibilidade: {min_sensitivity_simple_perceptron:.2f} - Especificidade: {min_specificity_simple_perceptron:.2f}')
    plt.figure(nfig)
    nfig += 1
    sns.heatmap(max_confusion_matrix_simple_perceptron, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f'Matriz de Confusão - Perceptron Simples (Maior Acurácia) - Acurácia: {max_accuracy_simple_perceptron:.2f} - Sensibilidade: {max_sensitivity_simple_perceptron:.2f} - Especificidade: {max_specificity_simple_perceptron:.2f}')
    plt.figure(nfig)
    nfig += 1
    sns.heatmap(min_confusion_matrix_adaline, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f'Matriz de Confusão - Adaline (Menor Acurácia) - Acurácia: {min_accuracy_adaline:.2f} - Sensibilidade: {min_sensitivity_adaline:.2f} - Especificidade: {min_specificity_adaline:.2f}')
    plt.figure(nfig)
    nfig += 1
    sns.heatmap(max_confusion_matrix_adaline, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f'Matriz de Confusão - Adaline (Maior Acurácia) - Acurácia: {max_accuracy_adaline:.2f} - Sensibilidade: {max_sensitivity_adaline:.2f} - Especificidade: {max_specificity_adaline:.2f}')
    
    plt.figure(nfig)
    nfig += 1
    sns.heatmap(min_confusion_matrix_mlp, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f'Matriz de Confusão - MLP (Menor Acurácia) - Acurácia: {min_accuracy_mlp:.2f} - Sensibilidade: {min_sensitivity_mlp:.2f} - Especificidade: {min_specificity_mlp:.2f}')
    plt.figure(nfig)
    nfig += 1
    sns.heatmap(max_confusion_matrix_mlp, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f'Matriz de Confusão - MLP (Maior Acurácia) - Acurácia: {max_accuracy_mlp:.2f} - Sensibilidade: {max_sensitivity_mlp:.2f} - Especificidade: {max_specificity_mlp:.2f}')

    plt.figure(nfig)
    nfig += 1
    max_simple_perceptron.plot_errors('Curva de Aprendizado - Perceptron Simples (Maior Acurácia)')
    plt.figure(nfig)
    nfig += 1
    min_simple_perceptron.plot_errors('Curva de Aprendizado - Perceptron Simples (Menor Acurácia)')
    plt.figure(nfig)
    nfig += 1
    max_adaline.plot_EQMs('Curva de Aprendizado - Adaline (Maior Acurácia)')
    plt.figure(nfig)
    nfig += 1
    min_adaline.plot_EQMs('Curva de Aprendizado - Adaline (Menor Acurácia)')
    plt.figure(nfig)
    nfig += 1
    max_mlp.plot_EQMs('Curva de Aprendizado - MLP (Maior Acurácia)')
    plt.figure(nfig)
    nfig += 1
    min_mlp.plot_EQMs('Curva de Aprendizado - MLP (Menor Acurácia)')
    plt.show()