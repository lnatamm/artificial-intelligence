# Imports
import numpy as np
import matplotlib.pyplot as plt
from modelos_RNA.adaline import Adaline
from modelos_RNA.multi_layer_perceptron import MLP
from config import Config

plot_graph = True

# Leitura dos dados
aerogerador = np.loadtxt("trabalhos\\Trabalho_Computacional_AV2\\datasets\\aerogerador.dat")

# Regressão Linear Simples

# Separação da variável dependente e independente
X = aerogerador[:, 0]  # Variável independente (primeira coluna)
y = aerogerador[:, 1]   # Variável dependente (segunda coluna)

if Config.REGRESSION_PLOT_GRAPH:
    plt.figure(1)
    plt.title('Dados Originais')
    plt.xlabel('Vento (m/s)')
    plt.ylabel('Potência (kW)')
    plt.scatter(X, y, color='blue', label='Dados Originais')

# Normalização dos dados
# X = (X - np.mean(X)) / np.std(X)
X = (X - np.min(X)) / (np.max(X) - np.min(X))
# y = (y - np.mean(y)) / np.std(y)
y = (y - np.min(y)) / (np.max(y) - np.min(y))
if Config.REGRESSION_PLOT_GRAPH:
    plt.figure(2)
    plt.title('Dados Normalizados')
    plt.xlabel('Vento (m/s)')
    plt.ylabel('Potência (kW)')
    plt.scatter(X, y, color='red', label='Dados Normalizados')

# Instanciação os modelo
adaline = Adaline(η=0.0001, ϵ=0.0001, epochs=1000)
mlp = MLP(input_size=1, q=[2, 1], m=1, η=0.0001, ϵ=0.0001, tolleration=10, epochs=Config.REGRESSION_EPOCHS)

eqms_adaline = []
r_s_adaline = []
eqms_mlp = []
r_s_mlp = []

for round in range(Config.REGRESSION_N_ROUNDS):
    print(f"Rodada: {round + 1}")
    # Aleatorização dos dados
    index = np.random.permutation(aerogerador.shape[0])
    X_shuffled = X[index]
    y_shuffled = y[index]
    # Separação dos dados em treino, teste e validação
    train_size = int(Config.REGRESSION_TRAIN * len(X))
    validation_size = int(Config.REGRESSION_VALIDATION * len(X))
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
    adaline.fit(X_train.reshape(-1, 1), y_train)
    # if Config.REGRESSION_PLOT_GRAPH:
    #     plt.figure(3)
    #     adaline.plot_EQMs()
    predictions_adaline = adaline.predict_regression(X_test.reshape(-1, 1))
    # MSE (Erro Quadrático Médio)
    EQM_adaline = np.mean((y_test - predictions_adaline) ** 2)
    # R² (coeficiente de determinação)
    ss_res_adaline = np.sum((y_test - predictions_adaline) ** 2)              # Soma dos quadrados dos resíduos
    ss_tot_adaline = np.sum((y_test - np.mean(y_test)) ** 2)          # Soma total dos quadrados
    r2_adaline = 1 - (ss_res_adaline / ss_tot_adaline)
    print(f"EQM: {EQM_adaline:.4f}")
    print(f"R²: {r2_adaline:.4f}")
    eqms_adaline.append(EQM_adaline)
    r_s_adaline.append(r2_adaline)

    # Multi-Layer Perceptron
    mlp.train_regression(X_train.reshape(-1, 1), y_train.reshape(-1, 1), X_validation.reshape(-1, 1), y_validation.reshape(-1, 1))
    predictions_mlp = []
    for x in X_test:
        x = np.vstack((-1, x))
        predictions_mlp.append(mlp.predict_regression(x.reshape(-1, 1)))
    EQM_mlp = np.mean((y_test - predictions_mlp) ** 2)
    ss_res_mlp = np.sum((y_test - predictions_mlp) ** 2)              # Soma dos quadrados dos resíduos
    ss_tot_mlp = np.sum((y_test - np.mean(y_test)) ** 2)          # Soma total dos quadrados
    r2_mlp = 1 - (ss_res_mlp / ss_tot_mlp)
    print(f"EQM: {EQM_mlp:.4f}")
    print(f"R²: {r2_mlp:.4f}")
    eqms_mlp.append(EQM_mlp)
    r_s_mlp.append(r2_mlp)
metrics_adaline = {
    "mean_eqm": np.mean(eqms_adaline),
    "std_eqm": np.std(eqms_adaline),
    "max_eqm": np.max(eqms_adaline),
    "min_eqm": np.min(eqms_adaline),
    "mean_r2": np.mean(r_s_adaline),
    "std_r2": np.std(r_s_adaline),
    "max_r2": np.max(r_s_adaline),
    "min_r2": np.min(r_s_adaline),
    "std_eqm": np.std(eqms_adaline),
}
metrics_mlp = {
    "mean_eqm": np.mean(eqms_mlp),
    "std_eqm": np.std(eqms_mlp),
    "max_eqm": np.max(eqms_mlp),
    "min_eqm": np.min(eqms_mlp),
    "mean_r2": np.mean(r_s_mlp),
    "std_r2": np.std(r_s_mlp),
    "max_r2": np.max(r_s_mlp),
    "min_r2": np.min(r_s_mlp),
}
models = [
    "Adaline",
    "MLP",
]

metrics = {
    "Média EQM": 
        (
            metrics_adaline["mean_eqm"],
            metrics_mlp["mean_eqm"],
        ),
    "Desvio Padrão EQM": 
        (
            metrics_adaline["max_eqm"],
            metrics_mlp["max_eqm"],
        ),
    "Máximo EQM": 
        (
            metrics_adaline["max_eqm"],
            metrics_mlp["max_eqm"],
        ),
    "Mínimo EQM": 
        (
            metrics_adaline["min_eqm"],
            metrics_mlp["min_eqm"],
        ),
    "Média R²": 
        (
            metrics_adaline["mean_r2"],
            metrics_mlp["mean_r2"],
        ),
    "Desvio Padrão R²": 
        (
            metrics_adaline["std_r2"],
            metrics_mlp["std_r2"],
        ),
    "Máximo R²": 
        (
            metrics_adaline["max_r2"],
            metrics_mlp["max_r2"],
        ),
    "Mínimo R²": 
        (
            metrics_adaline["min_r2"],
            metrics_mlp["min_r2"],
        ),
}

x = np.arange(len(models))  # the label locations
largura = 0.2  # the width of the bars
mult = 0

fig, ax = plt.subplots(layout='constrained')

for tipo, medida in metrics.items():
    offset = largura * mult
    rects = ax.bar(x + offset, medida, largura, label=tipo)
    ax.bar_label(rects, padding=3)
    mult += 1

ax.set_title('Métricas dos modelos')
ax.set_xticks(x + largura, models)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 2)

plt.show()