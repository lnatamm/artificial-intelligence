import numpy as np
import matplotlib.pyplot as plt

plot_graphs = True

# Tarefa de Regressão

# Configuração padrão dos plots
def get_plot_configuration(file, n_figure, title):
    fig = plt.figure(n_figure)
    plot = fig.add_subplot(projection='3d')

    plot.scatter(
        file[:, 0],
        file[:, 1],
        file[:, 2],
        color='teal',
        edgecolors='k'
    )

    plot.set_xlabel("Temperatura")
    plot.set_ylabel("pH da solução")
    plot.set_zlabel("Nível de Atividade Enzimática")
    plot.set_title(title)

    return plot

figure_index = 1

# 1.
atividade_enzimatica = np.loadtxt("datasets\\atividade_enzimatica.csv", delimiter=",")

N, p = atividade_enzimatica.shape[0], atividade_enzimatica.shape[1] - 1 # Removendo y

plot_1 = get_plot_configuration(atividade_enzimatica, figure_index, "Atividade Enzimática")
figure_index+=1

# 2.
X = atividade_enzimatica[:,:2]

y = atividade_enzimatica[:,-1]

# 3.
# Constantes
n_linspace = 40
x_axis = np.linspace(np.min(X[:,0]), np.max(X[:,0]), n_linspace)
y_axis = np.linspace(np.min(X[:,1]), np.max(X[:,1]), n_linspace)
X3d, Y3d = np.meshgrid(x_axis, y_axis)

# MQO Tradicional
plot_MQO_tradicional = get_plot_configuration(atividade_enzimatica, figure_index, "Atividade Enzimática - MQO Tradicional")
figure_index+=1

X_MQO_tradicional = np.hstack((
    np.ones((X.shape[0], 1)), X
))
B_MQO_tradicional = np.linalg.pinv(X_MQO_tradicional.T@X_MQO_tradicional)@X_MQO_tradicional.T@y

Z_MQO_tradicional = B_MQO_tradicional[0] + B_MQO_tradicional[1]*X3d + B_MQO_tradicional[2]* Y3d

plot_MQO_tradicional.plot_surface(X3d, Y3d, Z_MQO_tradicional, cmap='gray')

# MQO Regularizado
X_MQO_regularizado = np.hstack((
    np.ones((X.shape[0], 1)), X
))

# 4.
lambdas_MQO_regularizado = [0, 0.25, 0.5, 0.75, 1]

for lambda_MQO_regularizado in lambdas_MQO_regularizado:
    plot_MQO_regularizado = get_plot_configuration(atividade_enzimatica, figure_index, f"Atividade Enzimática - MQO Regularizado λ: {lambda_MQO_regularizado}")
    figure_index+=1
    B_MQO_regularizado = np.linalg.pinv(X_MQO_regularizado.T@X_MQO_regularizado + lambda_MQO_regularizado*np.identity(X_MQO_regularizado.shape[1]))@X_MQO_regularizado.T@y
    Z_MQO_regularizado = B_MQO_regularizado[0] + B_MQO_regularizado[1]*X3d + B_MQO_regularizado[2]*Y3d
    plot_MQO_regularizado.plot_surface(X3d, Y3d, Z_MQO_regularizado, cmap='gray')

# Média
plot_media = get_plot_configuration(atividade_enzimatica, figure_index, "Atividade Enzimática - Média")
figure_index+=1
media = np.mean(y)

B_media = [
    media,
    0,
    0
]

Z_media = B_media[0] + B_media[1]*X3d + B_media[2]*Y3d

plot_media.plot_surface(X3d, Y3d, Z_media, cmap='gray')

# 5.

# Simulações por Monte Carlo

rodadas = 500
particionamento = 0.8

desempenhos_MQO_tradicional = []
desempenhos_MQO_regularizado = []
desempenhos_media = []

for rodada in range(rodadas):
    index = np.random.permutation(atividade_enzimatica.shape[0])

    X_embaralhado = X[index, :]
    y_embaralhado = y[index]

    X_treino = X_embaralhado[:int(N*particionamento),:]
    y_treino = y_embaralhado[:int(N*particionamento)]

    X_teste = X_embaralhado[int(N*particionamento):,:]
    y_teste = y_embaralhado[int(N*particionamento):]

    X_treino = np.hstack((
        np.ones((X_treino.shape[0], 1)), X_treino
    ))

    X_teste = np.hstack((
        np.ones((X_teste.shape[0],1)), X_teste
    ))

    B_MQO_tradicional_MC = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@y_treino
    
    Bs_MQO_regularizado_MC = []

    for lambda_MQO_regularizado in lambdas_MQO_regularizado:
        B_MQO_regularizado_MC = np.linalg.pinv(X_treino.T@X_treino + lambda_MQO_regularizado*np.identity(X_treino.shape[1]))@X_treino.T@y_treino
        Bs_MQO_regularizado_MC.append(B_MQO_regularizado_MC)

    B_media_MC = [
        media,
        0,
        0
    ]

    y_predicao_MQO_tradicional = X_teste@B_MQO_tradicional_MC
    y_predicao_MQO_regularizado = [X_teste@Bs_MQO_regularizado_MC[i] for i in range(len(Bs_MQO_regularizado_MC))]
    y_predicao_media = X_teste@B_media_MC

    desempenhos_MQO_tradicional.append(np.sum((y_teste-y_predicao_MQO_tradicional)**2))
    desempenhos_MQO_regularizado.append(
        [
            np.sum((y_teste-y_predicao_MQO_regularizado[0])**2),
            np.sum((y_teste-y_predicao_MQO_regularizado[1])**2),
            np.sum((y_teste-y_predicao_MQO_regularizado[2])**2),
            np.sum((y_teste-y_predicao_MQO_regularizado[3])**2),
            np.sum((y_teste-y_predicao_MQO_regularizado[4])**2),
        ]
    )
    desempenhos_media.append(np.sum((y_teste-y_predicao_media)**2))

# 6.

# Média da variável dependente
metricas_media = {
    'media': np.mean(desempenhos_media),
    'desvio_padrao': np.std(desempenhos_media),
    'maximo': np.max(desempenhos_media),
    'minimo': np.min(desempenhos_media)
}
print("Média da variável dependente:")
print(f"Média: {metricas_media['media']}")
print(f"Desvio Padrão: {metricas_media['desvio_padrao']}")
print(f"Valor máximo: {metricas_media['maximo']}")
print(f"Valor mínimo: {metricas_media['minimo']}")
print("-------------------------------")

metricas_MQO_tradicional = {
    'media': np.mean(desempenhos_MQO_tradicional),
    'desvio_padrao': np.std(desempenhos_MQO_tradicional),
    'maximo': np.max(desempenhos_MQO_tradicional),
    'minimo': np.min(desempenhos_MQO_tradicional)
}
print("MQO tradicional:")
print(f"Média: {metricas_MQO_tradicional['media']}")
print(f"Desvio Padrão: {metricas_MQO_tradicional['desvio_padrao']}")
print(f"Valor máximo: {metricas_MQO_tradicional['maximo']}")
print(f"Valor mínimo: {metricas_MQO_tradicional['minimo']}")
print("-------------------------------")


desempenhos_MQO_regularizado = np.array(desempenhos_MQO_regularizado)

metricas_MQO_regularizado_025 = {
    'media': np.mean(desempenhos_MQO_regularizado[:,1]),
    'desvio_padrao': np.std(desempenhos_MQO_regularizado[:,1]),
    'maximo': np.max(desempenhos_MQO_regularizado[:,1]),
    'minimo': np.min(desempenhos_MQO_regularizado[:,1])
}
print("MQO regularizado (0,25):")
print(f"Média: {metricas_MQO_regularizado_025['media']}")
print(f"Desvio Padrão: {metricas_MQO_regularizado_025['desvio_padrao']}")
print(f"Valor máximo: {metricas_MQO_regularizado_025['maximo']}")
print(f"Valor mínimo: {metricas_MQO_regularizado_025['minimo']}")
print("-------------------------------")

metricas_MQO_regularizado_050 = {
    'media': np.mean(desempenhos_MQO_regularizado[:,2]),
    'desvio_padrao': np.std(desempenhos_MQO_regularizado[:,2]),
    'maximo': np.max(desempenhos_MQO_regularizado[:,2]),
    'minimo': np.min(desempenhos_MQO_regularizado[:,2])
}
print("MQO regularizado (0,5):")
print(f"Média: {metricas_MQO_regularizado_050['media']}")
print(f"Desvio Padrão: {metricas_MQO_regularizado_050['desvio_padrao']}")
print(f"Valor máximo: {metricas_MQO_regularizado_050['maximo']}")
print(f"Valor mínimo: {metricas_MQO_regularizado_050['minimo']}")
print("-------------------------------")

metricas_MQO_regularizado_075 = {
    'media': np.mean(desempenhos_MQO_regularizado[:,3]),
    'desvio_padrao': np.std(desempenhos_MQO_regularizado[:,3]),
    'maximo': np.max(desempenhos_MQO_regularizado[:,3]),
    'minimo': np.min(desempenhos_MQO_regularizado[:,3])
}
print("MQO regularizado (0,75):")
print(f"Média: {metricas_MQO_regularizado_075['media']}")
print(f"Desvio Padrão: {metricas_MQO_regularizado_075['desvio_padrao']}")
print(f"Valor máximo: {metricas_MQO_regularizado_075['maximo']}")
print(f"Valor mínimo: {metricas_MQO_regularizado_075['minimo']}")
print("-------------------------------")

metricas_MQO_regularizado_100 = {
    'media': np.mean(desempenhos_MQO_regularizado[:,4]),
    'desvio_padrao': np.std(desempenhos_MQO_regularizado[:,4]),
    'maximo': np.max(desempenhos_MQO_regularizado[:,4]),
    'minimo': np.min(desempenhos_MQO_regularizado[:,4])
}
print("MQO regularizado (1):")
print(f"Média: {metricas_MQO_regularizado_100['media']}")
print(f"Desvio Padrão: {metricas_MQO_regularizado_100['desvio_padrao']}")
print(f"Valor máximo: {metricas_MQO_regularizado_100['maximo']}")
print(f"Valor mínimo: {metricas_MQO_regularizado_100['minimo']}")
print("-------------------------------")

plt.figure(figure_index-1)

modelos = (
    "Média da variável dependente",
    "MQO tradicional",
    "MQO regularizado (0,25)",
    "MQO regularizado (0,5)",
    "MQO regularizado (0,75)",
    "MQO regularizado (1)"
)

metricas = {
    'Média': 
        (
            metricas_media['media'],
            metricas_MQO_tradicional['media'],
            metricas_MQO_regularizado_025['media'],
            metricas_MQO_regularizado_050['media'],
            metricas_MQO_regularizado_075['media'],
            metricas_MQO_regularizado_100['media']
        ),
    'Desvio Padrão': 
        (
            metricas_media['desvio_padrao'],
            metricas_MQO_tradicional['desvio_padrao'],
            metricas_MQO_regularizado_025['desvio_padrao'],
            metricas_MQO_regularizado_050['desvio_padrao'],
            metricas_MQO_regularizado_075['desvio_padrao'],
            metricas_MQO_regularizado_100['desvio_padrao']
        ),
    'Valor máximo': 
        (
            metricas_media['maximo'],
            metricas_MQO_tradicional['maximo'],
            metricas_MQO_regularizado_025['maximo'],
            metricas_MQO_regularizado_050['maximo'],
            metricas_MQO_regularizado_075['maximo'],
            metricas_MQO_regularizado_100['maximo']
        ),
    'Valor mínimo': 
        (
            metricas_media['minimo'],
            metricas_MQO_tradicional['minimo'],
            metricas_MQO_regularizado_025['minimo'],
            metricas_MQO_regularizado_050['minimo'],
            metricas_MQO_regularizado_075['minimo'],
            metricas_MQO_regularizado_100['minimo']
        ),
}

x = np.arange(len(modelos))  # the label locations
largura = 0.2  # the width of the bars
mult = 0

fig, ax = plt.subplots(layout='constrained')

for tipo, medida in metricas.items():
    offset = largura * mult
    rects = ax.bar(x + offset, medida, largura, label=tipo)
    ax.bar_label(rects, padding=3)
    mult += 1

ax.set_title('Métricas dos modelos')
ax.set_xticks(x + largura, modelos)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 40)


if plot_graphs:
    plt.show()