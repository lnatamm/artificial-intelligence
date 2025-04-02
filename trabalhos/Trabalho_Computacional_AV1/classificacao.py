import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plot_graphs = True

# Constantes

classes = [1, 2, 3, 4, 5]

classes_name = [
    "Neutro",
    "Sorrindo",
    "Sobrancelhas levantadas",
    "Surpreso",
    "Rabugento"
]

colors = [
    "gray",
    "y",
    "green",
    "teal",
    "red",
]

cmap = ListedColormap(colors)

# Índice para a figure

index_plot = 1

# Configuração padrão dos plots
def get_plot_configuration(n_figure, title, dados_classes, classes, colors):
    fig = plt.figure(n_figure)
    plot = fig.add_subplot()
    for i in range(len(classes)):
        plot.scatter(x=dados_classes[i][0, :],y=dados_classes[i][1, :],label=classes_name[i],
            c=colors[i],ec="k")

    plot.set_xlabel("Corrugador do Supercílio")
    plot.set_ylabel("Zigomático Maior")
    plot.legend()
    plot.set_title(title)

    return plot


# Função para plotar a fronteira de decisão do MQO
def plot_frontier_MQO(plot, W):
    x_axis = np.linspace(-200,5000,1000)
    x2 = -W[0,0]/W[2,0] - W_MQO[1,0]/W[2,0]*x_axis
    plot.plot(x_axis,x2,'--k')
    x2 = -W[0,1]/W[2,1] - W[1,1]/W[2,1]*x_axis
    plot.plot(x_axis,x2,'--k')
    x2 = -W[0,2]/W[2,2] - W[1,2]/W[2,2]*x_axis
    plot.plot(x_axis,x2,'--k')
    x2 = -W[0,3]/W[2,3] - W[1,3]/W[2,3]*x_axis
    plot.plot(x_axis,x2,'--k')
    x2 = -W[0,4]/W[2,4] - W[1,4]/W[2,4]*x_axis
    plot.plot(x_axis,x2,'--k')

    x3d , y3d = np.meshgrid(x_axis,x_axis)
    x1 = np.ravel(x3d)
    x2 = np.ravel(y3d)

    X_plot = np.hstack((
        np.ones((len(x1),1)),
        x1.reshape(len(x1),1),
        x2.reshape(len(x1),1),
    ))

    y_predicao = X_plot@W
    y_plot = np.argmax(y_predicao,axis=1)
    y_plot = y_plot.reshape(x3d.shape)
    plot.contourf(x3d,y3d,y_plot,alpha=.3,cmap=cmap, levels=np.arange(-0.5, 5.5, 1))
    plot.set_xlim(-100,4170)
    plot.set_ylim(-100,4170)
    return plot


# Função para plotar a fronteira de decisão dos modelos gaussianos bayesianos que utilizam a função discriminante da distância de mahalanois
def plot_frontier_Gauss_mahalanobis(plot, medias, matriz_de_covariancia_inversa):
    # Criar grid de pontos para cobrir a área dos dados
    x_min, x_max = X_Gauss[0, :].min() - 500, X_Gauss[0, :].max() + 500
    y_min, y_max = X_Gauss[1, :].min() - 500, X_Gauss[1, :].max() + 500

    # Criar a malha de pontos (grid)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),
                        np.linspace(y_min, y_max, 1000))

    # Vetorizar os pontos para facilitar a computação
    X_grid = np.c_[xx.ravel(), yy.ravel()]  # (10000, 2) pontos

    # Calcular a distância de mahalanois para cada classe
    Z = []
    for x in X_grid:
        distancias_de_mahalanobis = [funcao_discriminante_cov_igual(x, medias[i], matriz_de_covariancia_inversa) for i in range(len(classes))]
        Z.append(np.argmin(distancias_de_mahalanobis))  # Pegamos a classe com menor distância de mahalanobis
    

    # Converter para formato de grid correto
    Z = np.array(Z).reshape(xx.shape)  # Agora Z tem o mesmo formato de xx e yy (100,100)
    unique, counts = np.unique(Z, return_counts=True)
    print("Distribuição das classes na malha:", dict(zip(unique, counts)))
    plot.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.arange(-0.5, 5.5, 1))
    plot.set_xlim(-100,4170)
    plot.set_ylim(-100,4170)
    return plot

# Função para plotar a fronteira de decisão dos modelos gaussianos bayesianos que utilizam a função discriminante de friedman para λ != 1
def plot_frontier_Gauss_friedman(plot, medias, matrizes_de_covariancia, matrizes_de_covariancia_inversas):
    # Criar grid de pontos para cobrir a área dos dados
    x_min, x_max = X_Gauss[0, :].min() - 500, X_Gauss[0, :].max() + 500
    y_min, y_max = X_Gauss[1, :].min() - 500, X_Gauss[1, :].max() + 500

    # Criar a malha de pontos (grid)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),
                        np.linspace(y_min, y_max, 1000))

    # Vetorizar os pontos para facilitar a computação
    X_grid = np.c_[xx.ravel(), yy.ravel()]  # (10000, 2) pontos
    Z = []
    for x in X_grid:
        vetor_friedman = [funcao_discriminante_friedman(x, medias[i], matrizes_de_covariancia[i], matrizes_de_covariancia_inversas[i]) for i in range(len(classes))]
        Z.append(np.argmax(vetor_friedman))  # Pegamos o argmax do vetor de friedman
    Z = np.array(Z).reshape(xx.shape)  # Agora Z tem o mesmo formato de xx e yy (100,100)
    plot.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.arange(-0.5, 5.5, 1))
    plot.set_xlim(-100,4170)
    plot.set_ylim(-100,4170)
    return plot

# Função para plotar a fronteira de decisão dos modelos gaussianos bayesianos que utilizam a FDP como função discriminante
def plot_frontier_Gauss(plot, medias, matrizes_de_covariancia, matrizes_de_covariancia_inversas):
    # Criar grid de pontos para cobrir a área dos dados
    x_min, x_max = X_Gauss[0, :].min() - 500, X_Gauss[0, :].max() + 500
    y_min, y_max = X_Gauss[1, :].min() - 500, X_Gauss[1, :].max() + 500

    # Criar a malha de pontos (grid)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),
                        np.linspace(y_min, y_max, 1000))

    # Vetorizar os pontos para facilitar a computação
    X_grid = np.c_[xx.ravel(), yy.ravel()]  # (10000, 2) pontos
    Z = []
    for x in X_grid:
        probabilidades = [fdp(x, medias[i], matrizes_de_covariancia[i], matrizes_de_covariancia_inversas[i]) for i in range(len(classes))]
        Z.append(np.argmax(probabilidades))
    Z = np.array(Z).reshape(xx.shape)  # Agora Z tem o mesmo formato de xx e yy (100,100)
    plot.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.arange(-0.5, 5.5, 1))
    plot.set_xlim(-100,4170)
    plot.set_ylim(-100,4170)
    return plot 

def acuracia_MQO(X, y, W, N):
    acertos = 0
    for i, x in enumerate(X):
        if np.argmax(x@W) == np.argmax(y[i]):
            acertos += 1
    return (acertos/N) * 100

def acuracia_mahalanois(X, y, medias, matriz_de_covariancia_inversa, N):
    acertos = 0
    for i, x in enumerate(X.T):
        distancia_de_mahalanois = []
        for j, classe in enumerate(classes):
            distancia_de_mahalanois.append(funcao_discriminante_cov_igual(x, medias[j], matriz_de_covariancia_inversa))
        if np.argmin(distancia_de_mahalanois) == np.argmax(y[i]):
            acertos += 1
    return (acertos/N) * 100

def acuracia_friedman(X, y, medias, matrizes_de_covariancia, matrizes_de_covariancia_inversas, N):
    acertos = 0
    for i, x in enumerate(X.T):
        distancia_de_mahalanois = []
        for j, classe in enumerate(classes):
            distancia_de_mahalanois.append(funcao_discriminante_friedman(x, medias[j], matrizes_de_covariancia[j], matrizes_de_covariancia_inversas[j]))
        if np.argmax(distancia_de_mahalanois) == np.argmax(y[i]):
            acertos += 1
    return (acertos/N) * 100

def acuracia_Gauss(X, y, medias, matrizes_de_covariancia, matrizes_de_covariancia_inversas, N):
    acertos = 0
    for i, x in enumerate(X.T):
        distancia_de_mahalanois = []
        for j, classe in enumerate(classes):
            distancia_de_mahalanois.append(fdp(x, medias[j], matrizes_de_covariancia[j], matrizes_de_covariancia_inversas[j]))
        if np.argmax(distancia_de_mahalanois) == np.argmax(y[i]):
            acertos += 1
    return (acertos/N) * 100
figure_index = 1

def fdp(x, mu, sigma, sigma_inv):
    p = len(mu)  # Dimensão do vetor
    coef = -0.5 * (p * np.log(2 * np.pi) + np.log(np.linalg.det(sigma)))
    quad_form = -0.5 * (x - mu).T @ sigma_inv @ (x - mu)
    
    return coef + quad_form

def funcao_discriminante_cov_igual(x, mu, sigma_inv):
    return (x - mu).T@sigma_inv@ (x - mu)

def funcao_discriminante_friedman(x, media, matriz_de_covariancia, matriz_de_covariancia_inversa):
    return -0.5 * np.log(np.linalg.det(matriz_de_covariancia)) - np.dot(0.5, (x - media).T) @ matriz_de_covariancia_inversa @ (x - media)

# 1.

EMGs = np.loadtxt("datasets\\EMGsDataset.csv", delimiter=',')
EMGs_MQO = EMGs.T

dados_classes = [
    EMGs[:2,EMGs[-1,:] == classes[0]],
    EMGs[:2,EMGs[-1,:] == classes[1]],
    EMGs[:2,EMGs[-1,:] == classes[2]],
    EMGs[:2,EMGs[-1,:] == classes[3]],
    EMGs[:2,EMGs[-1,:] == classes[4]]
]

N, p = EMGs.T.shape[0], EMGs.shape[1] - 1 # Removendo y


# Todas as linhas da primeira até a segunda e todas as colunas onde a última linha é igual a cada classe respectiva.
# Isso organiza os dados para ser primeiro da classe 1, 2, 3... c
X_Gauss = np.hstack((
    dados_classes[0],
    dados_classes[1],
    dados_classes[2],
    dados_classes[3],
    dados_classes[4],
))

y_Gauss = np.vstack((
    np.tile(np.array([[1, -1, -1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, 1, -1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, 1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, -1, 1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, -1, -1, 1]]), (10000, 1)),
))

# LMQ (MQO)

X_MQO = np.vstack((
    EMGs_MQO[EMGs_MQO[:,-1] == classes[0],:2],
    EMGs_MQO[EMGs_MQO[:,-1] == classes[1],:2],
    EMGs_MQO[EMGs_MQO[:,-1] == classes[2],:2],
    EMGs_MQO[EMGs_MQO[:,-1] == classes[3],:2],
    EMGs_MQO[EMGs_MQO[:,-1] == classes[4],:2],
))

X_MQO = np.hstack((
    np.ones((N, 1)), X_MQO
))

y_MQO = np.vstack((
    np.tile(np.array([[1, -1, -1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, 1, -1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, 1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, -1, 1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, -1, -1, 1]]), (10000, 1)),
))

W_MQO = np.linalg.pinv(X_MQO.T@X_MQO)@X_MQO.T@y_MQO

# 2/3.

if plot_graphs:
    plot = get_plot_configuration(index_plot, "MQO", dados_classes, classes, colors)
    x_novo = np.hstack((1, EMGs[np.random.randint(0, len(EMGs)), 0:2]))
    plot = plot_frontier_MQO(plot, W_MQO)
    index_plot += 1

# Gaussiano Bayesiano Tradicional
matrizes_de_covariancia = np.array([
    np.cov(dados_classes[0]),
    np.cov(dados_classes[1]),
    np.cov(dados_classes[2]),
    np.cov(dados_classes[3]),
    np.cov(dados_classes[4]),
])
matrizes_de_covariancia_inversas = np.array([
    np.linalg.pinv(matrizes_de_covariancia[0]),
    np.linalg.pinv(matrizes_de_covariancia[1]),
    np.linalg.pinv(matrizes_de_covariancia[2]),
    np.linalg.pinv(matrizes_de_covariancia[3]),
    np.linalg.pinv(matrizes_de_covariancia[4]),
])
medias = np.array([
    np.mean(dados_classes[0], axis=1),
    np.mean(dados_classes[1], axis=1),
    np.mean(dados_classes[2], axis=1),
    np.mean(dados_classes[3], axis=1),
    np.mean(dados_classes[4], axis=1),
])

if plot_graphs:
    plot = get_plot_configuration(index_plot, "Gaussiano Bayesiano (Tradicional)", dados_classes, classes, colors)
    plot = plot_frontier_Gauss(plot, medias, matrizes_de_covariancia, matrizes_de_covariancia_inversas)
    index_plot += 1

# Gaussiano Bayesiano Covariancia Igual
matriz_de_covariancia = np.cov(X_Gauss)
matriz_de_covariancia_inversa = np.linalg.pinv(matriz_de_covariancia)
g_covariancia_igual = []

if plot_graphs:
    plot = get_plot_configuration(index_plot, "Gaussiano Bayesiano (Cov. Igual)", dados_classes, classes, colors)
    plot = plot_frontier_Gauss_mahalanobis(plot, medias, matriz_de_covariancia_inversa)
    index_plot += 1 

matriz_de_covariancia_agregada = np.sum([matrizes_de_covariancia[i]*dados_classes[i].shape[1]/EMGs.shape[1] for i in range(5)], axis=0)

matriz_de_covariancia_agregada_inversa = np.linalg.pinv(matriz_de_covariancia_agregada)
plot = get_plot_configuration(index_plot, "Gaussiano Bayesiano (Cov. Agregada)", dados_classes, classes, colors)
plot = plot_frontier_Gauss_mahalanobis(plot, medias, matriz_de_covariancia_agregada_inversa)
index_plot += 1


# 4.
# Gaussiano Bayesiano Friedman
lambdas = [0.25, 0.50, 0.75]

matrizes_de_covariancia_025 = [((1-lambdas[0])*dados_classes[i].shape[1]*np.cov(dados_classes[i]) + N*lambdas[0]*matriz_de_covariancia_agregada)/((1-lambdas[0])*dados_classes[i].shape[1] + N*lambdas[0]) for i in range(5)]

matrizes_de_covariancia_025_inversas = [
    np.linalg.pinv(matrizes_de_covariancia_025[0]),
    np.linalg.pinv(matrizes_de_covariancia_025[1]),
    np.linalg.pinv(matrizes_de_covariancia_025[2]),
    np.linalg.pinv(matrizes_de_covariancia_025[3]),
    np.linalg.pinv(matrizes_de_covariancia_025[4]),
]

if plot_graphs:
    plot = get_plot_configuration(index_plot, "Gaussiano Bayesiano Friedman (025)", dados_classes, classes, colors)
    plot = plot_frontier_Gauss_friedman(plot, medias, matrizes_de_covariancia_025, matrizes_de_covariancia_025_inversas)
    index_plot += 1

matrizes_de_covariancia_050 = [((1-lambdas[1])*dados_classes[i].shape[1]*np.cov(dados_classes[i]) + N*lambdas[1]*matriz_de_covariancia_agregada)/((1-lambdas[1])*dados_classes[i].shape[1] + N*lambdas[1]) for i in range(5)]
matrizes_de_covariancia_050_inversas = [
    np.linalg.pinv(matrizes_de_covariancia_050[0]),
    np.linalg.pinv(matrizes_de_covariancia_050[1]),
    np.linalg.pinv(matrizes_de_covariancia_050[2]),
    np.linalg.pinv(matrizes_de_covariancia_050[3]),
    np.linalg.pinv(matrizes_de_covariancia_050[4]),
]

if plot_graphs:
    plot = get_plot_configuration(index_plot, "Gaussiano Bayesiano Friedman (050)", dados_classes, classes, colors)
    plot = plot_frontier_Gauss_friedman(plot, medias, matrizes_de_covariancia_050, matrizes_de_covariancia_050_inversas)
    index_plot += 1

matrizes_de_covariancia_075 = [((1-lambdas[2])*dados_classes[i].shape[1]*np.cov(dados_classes[i]) + N*lambdas[2]*matriz_de_covariancia_agregada)/((1-lambdas[2])*dados_classes[i].shape[1] + N*lambdas[2]) for i in range(5)]
matrizes_de_covariancia_075_inversas = [
    np.linalg.pinv(matrizes_de_covariancia_075[0]),
    np.linalg.pinv(matrizes_de_covariancia_075[1]),
    np.linalg.pinv(matrizes_de_covariancia_075[2]),
    np.linalg.pinv(matrizes_de_covariancia_075[3]),
    np.linalg.pinv(matrizes_de_covariancia_075[4]),
]

if plot_graphs:
    plot = get_plot_configuration(index_plot, "Gaussiano Bayesiano Friedman (075)", dados_classes, classes, colors)
    plot = plot_frontier_Gauss_friedman(plot, medias, matrizes_de_covariancia_075, matrizes_de_covariancia_075_inversas)
    index_plot += 1

matrizes_de_covariancia_naive = [
    np.diag([np.std(EMGs[0, :]), np.std(EMGs[1, :])]),
    np.diag([np.std(EMGs[0, :]), np.std(EMGs[1, :])]),
    np.diag([np.std(EMGs[0, :]), np.std(EMGs[1, :])]),
    np.diag([np.std(EMGs[0, :]), np.std(EMGs[1, :])]),
    np.diag([np.std(EMGs[0, :]), np.std(EMGs[1, :])]),
]

matrizes_de_covariancia_naive_inversas = [
    np.linalg.pinv(matrizes_de_covariancia_naive[0]),
    np.linalg.pinv(matrizes_de_covariancia_naive[1]),
    np.linalg.pinv(matrizes_de_covariancia_naive[2]),
    np.linalg.pinv(matrizes_de_covariancia_naive[3]),
    np.linalg.pinv(matrizes_de_covariancia_naive[4]),
]

if plot_graphs:
    plot = get_plot_configuration(index_plot, "Gaussiano Bayesiano Naive", dados_classes, classes, colors)
    plot = plot_frontier_Gauss(plot, medias, matrizes_de_covariancia_naive, matrizes_de_covariancia_naive_inversas)
    index_plot += 1

# 5.
rodadas = 500
particionamento = 0.8

desempenhos_MQO_tradicional = []
desempenhos_gaussiano_tradicional = []
desempenhos_gaussiano_cov_igual = []
desempenhos_gaussiano_cov_agregada = []
desempenhos_naive_bayes = []
desempenhos_gaussiano_regularizado = []

for rodada in range(rodadas):
    print(f"Rodada: {rodada+1}")

    # Indices permutados
    index = np.random.permutation(N)

    # LMQ (MQO Linear)
    X_MQO_embaralhado = X_MQO[index, :]
    y_MQO_embaralhado = y_MQO[index]

    X_treino_MQO = X_MQO_embaralhado[:int(N*particionamento),:]
    y_treino_MQO = y_MQO_embaralhado[:int(N*particionamento)]

    X_teste_MQO = X_MQO_embaralhado[int(N*particionamento):,:]
    y_teste_MQO = y_MQO_embaralhado[int(N*particionamento):]

    W_MQO_tradicional_MC = np.linalg.pinv(X_treino_MQO.T@X_treino_MQO)@X_treino_MQO.T@y_treino_MQO

    desempenhos_MQO_tradicional.append(acuracia_MQO(X_teste_MQO, y_teste_MQO, W_MQO_tradicional_MC, len(y_teste_MQO)))

    # Gaussiano Bayesiano Tradicional
    dados = EMGs.copy()
    X_Gauss_embaralhado = X_Gauss[:, index]
    y_Gauss_embaralhado = y_Gauss[index]
    dados_embaralhado = dados[:, index]

    X_treino_Gauss = X_Gauss_embaralhado[:,:int(N*particionamento)]
    y_treino_Gauss = y_Gauss_embaralhado[:int(N*particionamento)]
    dados_treino = dados_embaralhado[:, :int(N*particionamento)]

    X_teste_Gauss = X_Gauss_embaralhado[:,int(N*particionamento):]
    y_teste_Gauss = y_Gauss_embaralhado[int(N*particionamento):]
    dados_teste = dados_embaralhado[:, int(N*particionamento):]

    dados_classes = [
        dados_treino[:2,dados_treino[-1,:] == classes[0]],
        dados_treino[:2,dados_treino[-1,:] == classes[1]],
        dados_treino[:2,dados_treino[-1,:] == classes[2]],
        dados_treino[:2,dados_treino[-1,:] == classes[3]],
        dados_treino[:2,dados_treino[-1,:] == classes[4]]
    ]
    

    medias = np.array([
        np.mean(dados_classes[0], axis=1),
        np.mean(dados_treino[:2,dados_treino[-1,:] == classes[1]], axis=1),
        np.mean(dados_treino[:2,dados_treino[-1,:] == classes[2]], axis=1),
        np.mean(dados_treino[:2,dados_treino[-1,:] == classes[3]], axis=1),
        np.mean(dados_treino[:2,dados_treino[-1,:] == classes[4]], axis=1),
    ])
    try:
        matrizes_de_covariancia = np.array([
            np.cov(dados_classes[0]),
            np.cov(dados_classes[1]),
            np.cov(dados_classes[2]),
            np.cov(dados_classes[3]),
            np.cov(dados_classes[4]),
        ])
        matrizes_de_covariancia_inversas = np.array([
            np.linalg.pinv(matrizes_de_covariancia[0]),
            np.linalg.pinv(matrizes_de_covariancia[1]),
            np.linalg.pinv(matrizes_de_covariancia[2]),
            np.linalg.pinv(matrizes_de_covariancia[3]),
            np.linalg.pinv(matrizes_de_covariancia[4]),
        ])
        desempenhos_gaussiano_tradicional.append(acuracia_Gauss(X_teste_Gauss, y_teste_Gauss, medias, matrizes_de_covariancia, matrizes_de_covariancia_inversas, len(y_teste_Gauss)))
    except Exception as e:
        print("Erro")
        print(e)

    
    
    # Gaussiano Bayesiano Covariancia Igual
    matriz_de_covariancia = np.cov(X_treino_Gauss)
    matriz_de_covariancia_inversa = np.linalg.pinv(matriz_de_covariancia)
    desempenhos_gaussiano_cov_igual.append(acuracia_mahalanois(X_teste_Gauss, y_teste_Gauss, medias, matriz_de_covariancia_inversa, len(y_teste_Gauss)))

    # Gaussiano Bayesiano Covariancia Agregada
    
    matrizes_de_covariancia = np.array([
        np.cov(dados_classes[0]),
        np.cov(dados_treino[:2,dados_treino[-1,:] == classes[1]]),
        np.cov(dados_treino[:2,dados_treino[-1,:] == classes[2]]),
        np.cov(dados_treino[:2,dados_treino[-1,:] == classes[3]]),
        np.cov(dados_treino[:2,dados_treino[-1,:] == classes[4]]),
    ])

    matriz_de_covariancia_agregada = np.sum([matrizes_de_covariancia[i]*dados_classes[i].shape[1]/dados_treino.shape[1] for i in range(5)], axis=0)

    matriz_de_covariancia_agregada_inversa = np.linalg.pinv(matriz_de_covariancia_agregada)
    desempenhos_gaussiano_cov_agregada.append(acuracia_mahalanois(X_teste_Gauss, y_teste_Gauss, medias, matriz_de_covariancia_agregada_inversa, len(y_teste_Gauss)))

    # Gaussiano Bayesiano Friedman
    lambdas = [0.25, 0.50, 0.75]

    matrizes_de_covariancia = np.array([
        np.cov(dados_classes[0]),
        np.cov(dados_classes[1]),
        np.cov(dados_classes[2]),
        np.cov(dados_classes[3]),
        np.cov(dados_classes[4]),
    ])

    matrizes_de_covariancia_025 = [((1-lambdas[0])*dados_classes[i].shape[1]*matrizes_de_covariancia[i] + dados_treino.shape[1]*lambdas[0]*matriz_de_covariancia_agregada)/((1-lambdas[0])*dados_classes[i].shape[1] + dados_treino.shape[1]*lambdas[0]) for i in range(5)]

    matrizes_de_covariancia_025_inversas = [
        np.linalg.pinv(matrizes_de_covariancia_025[0]),
        np.linalg.pinv(matrizes_de_covariancia_025[1]),
        np.linalg.pinv(matrizes_de_covariancia_025[2]),
        np.linalg.pinv(matrizes_de_covariancia_025[3]),
        np.linalg.pinv(matrizes_de_covariancia_025[4]),
    ]

    matrizes_de_covariancia_050 = [((1-lambdas[1])*dados_classes[i].shape[1]*matrizes_de_covariancia[i] + dados_treino.shape[1]*lambdas[1]*matriz_de_covariancia_agregada)/((1-lambdas[1])*dados_classes[i].shape[1] + dados_treino.shape[1]*lambdas[1]) for i in range(5)]
    matrizes_de_covariancia_050_inversas = [
        np.linalg.pinv(matrizes_de_covariancia_050[0]),
        np.linalg.pinv(matrizes_de_covariancia_050[1]),
        np.linalg.pinv(matrizes_de_covariancia_050[2]),
        np.linalg.pinv(matrizes_de_covariancia_050[3]),
        np.linalg.pinv(matrizes_de_covariancia_050[4]),
    ]
    
    matrizes_de_covariancia_075 = [((1-lambdas[2])*dados_classes[i].shape[1]*matrizes_de_covariancia[i] + dados_treino.shape[1]*lambdas[2]*matriz_de_covariancia_agregada)/((1-lambdas[2])*dados_classes[i].shape[1] + dados_treino.shape[1]*lambdas[2]) for i in range(5)]
    matrizes_de_covariancia_075_inversas = [
        np.linalg.pinv(matrizes_de_covariancia_075[0]),
        np.linalg.pinv(matrizes_de_covariancia_075[1]),
        np.linalg.pinv(matrizes_de_covariancia_075[2]),
        np.linalg.pinv(matrizes_de_covariancia_075[3]),
        np.linalg.pinv(matrizes_de_covariancia_075[4]),
    ]

    desempenhos_gaussiano_regularizado.append(
        [
            acuracia_friedman(X_teste_Gauss, y_teste_Gauss, medias, matrizes_de_covariancia_025, matrizes_de_covariancia_025_inversas, len(y_teste_Gauss)),
            acuracia_friedman(X_teste_Gauss, y_teste_Gauss, medias, matrizes_de_covariancia_050, matrizes_de_covariancia_050_inversas, len(y_teste_Gauss)),
            acuracia_friedman(X_teste_Gauss, y_teste_Gauss, medias, matrizes_de_covariancia_075, matrizes_de_covariancia_075_inversas, len(y_teste_Gauss)),
        ]
    )

    # Gaussiano Bayesiano Naive Bayes
    matrizes_de_covariancia_naive = [
        np.diag([np.std(EMGs[0, :]), np.std(EMGs[1, :])]),
        np.diag([np.std(EMGs[0, :]), np.std(EMGs[1, :])]),
        np.diag([np.std(EMGs[0, :]), np.std(EMGs[1, :])]),
        np.diag([np.std(EMGs[0, :]), np.std(EMGs[1, :])]),
        np.diag([np.std(EMGs[0, :]), np.std(EMGs[1, :])]),
    ]

    matrizes_de_covariancia_naive_inversas = [
        np.linalg.pinv(matrizes_de_covariancia_naive[0]),
        np.linalg.pinv(matrizes_de_covariancia_naive[1]),
        np.linalg.pinv(matrizes_de_covariancia_naive[2]),
        np.linalg.pinv(matrizes_de_covariancia_naive[3]),
        np.linalg.pinv(matrizes_de_covariancia_naive[4]),
    ]

    desempenhos_naive_bayes.append(acuracia_Gauss(X_teste_Gauss, y_teste_Gauss, medias, matrizes_de_covariancia_naive, matrizes_de_covariancia_naive_inversas, len(y_teste_Gauss)))


# 6.
desempenhos_MQO_tradicional = np.array(desempenhos_MQO_tradicional)
desempenhos_gaussiano_tradicional = np.array(desempenhos_gaussiano_tradicional)
desempenhos_gaussiano_cov_igual = np.array(desempenhos_gaussiano_cov_igual)
desempenhos_gaussiano_cov_agregada = np.array(desempenhos_gaussiano_cov_agregada)
desempenhos_naive_bayes = np.array(desempenhos_naive_bayes)
desempenhos_gaussiano_regularizado = np.array(desempenhos_gaussiano_regularizado)

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

metricas_Gauss_tradicional = {
    'media': np.mean(desempenhos_gaussiano_tradicional),
    'desvio_padrao': np.std(desempenhos_gaussiano_tradicional),
    'maximo': np.max(desempenhos_gaussiano_tradicional),
    'minimo': np.min(desempenhos_gaussiano_tradicional)
}
print("Gauss (Tradicional):")
print(f"Média: {metricas_Gauss_tradicional['media']}")
print(f"Desvio Padrão: {metricas_Gauss_tradicional['desvio_padrao']}")
print(f"Valor máximo: {metricas_Gauss_tradicional['maximo']}")
print(f"Valor mínimo: {metricas_Gauss_tradicional['minimo']}")
print("-------------------------------")

metricas_Gauss_cov_igual = {
    'media': np.mean(desempenhos_gaussiano_cov_igual),
    'desvio_padrao': np.std(desempenhos_gaussiano_cov_igual),
    'maximo': np.max(desempenhos_gaussiano_cov_igual),
    'minimo': np.min(desempenhos_gaussiano_cov_igual)
}
print("Gauss (Cov. Igual):")
print(f"Média: {metricas_Gauss_cov_igual['media']}")
print(f"Desvio Padrão: {metricas_Gauss_cov_igual['desvio_padrao']}")
print(f"Valor máximo: {metricas_Gauss_cov_igual['maximo']}")
print(f"Valor mínimo: {metricas_Gauss_cov_igual['minimo']}")
print("-------------------------------")

metricas_Gauss_cov_agregada = {
    'media': np.mean(desempenhos_gaussiano_cov_agregada),
    'desvio_padrao': np.std(desempenhos_gaussiano_cov_agregada),
    'maximo': np.max(desempenhos_gaussiano_cov_agregada),
    'minimo': np.min(desempenhos_gaussiano_cov_agregada)
}
print("Gauss (Cov. Agregada):")
print(f"Média: {metricas_Gauss_cov_agregada['media']}")
print(f"Desvio Padrão: {metricas_Gauss_cov_agregada['desvio_padrao']}")
print(f"Valor máximo: {metricas_Gauss_cov_agregada['maximo']}")
print(f"Valor mínimo: {metricas_Gauss_cov_agregada['minimo']}")
print("-------------------------------")

metricas_Gauss_regularizado_025 = {
    'media': np.mean(desempenhos_gaussiano_regularizado[:,0]),
    'desvio_padrao': np.std(desempenhos_gaussiano_regularizado[:,0]),
    'maximo': np.max(desempenhos_gaussiano_regularizado[:,0]),
    'minimo': np.min(desempenhos_gaussiano_regularizado[:,0])
}
print("Gauss regularizado (0,25):")
print(f"Média: {metricas_Gauss_regularizado_025['media']}")
print(f"Desvio Padrão: {metricas_Gauss_regularizado_025['desvio_padrao']}")
print(f"Valor máximo: {metricas_Gauss_regularizado_025['maximo']}")
print(f"Valor mínimo: {metricas_Gauss_regularizado_025['minimo']}")
print("-------------------------------")

metricas_Gauss_regularizado_050 = {
    'media': np.mean(desempenhos_gaussiano_regularizado[:,1]),
    'desvio_padrao': np.std(desempenhos_gaussiano_regularizado[:,1]),
    'maximo': np.max(desempenhos_gaussiano_regularizado[:,1]),
    'minimo': np.min(desempenhos_gaussiano_regularizado[:,1])
}
print("Gauss regularizado (0,50):")
print(f"Média: {metricas_Gauss_regularizado_050['media']}")
print(f"Desvio Padrão: {metricas_Gauss_regularizado_050['desvio_padrao']}")
print(f"Valor máximo: {metricas_Gauss_regularizado_050['maximo']}")
print(f"Valor mínimo: {metricas_Gauss_regularizado_050['minimo']}")
print("-------------------------------")

metricas_Gauss_regularizado_075 = {
    'media': np.mean(desempenhos_gaussiano_regularizado[:,2]),
    'desvio_padrao': np.std(desempenhos_gaussiano_regularizado[:,2]),
    'maximo': np.max(desempenhos_gaussiano_regularizado[:,2]),
    'minimo': np.min(desempenhos_gaussiano_regularizado[:,2])
}
print("Gauss regularizado (0,75):")
print(f"Média: {metricas_Gauss_regularizado_075['media']}")
print(f"Desvio Padrão: {metricas_Gauss_regularizado_075['desvio_padrao']}")
print(f"Valor máximo: {metricas_Gauss_regularizado_075['maximo']}")
print(f"Valor mínimo: {metricas_Gauss_regularizado_075['minimo']}")
print("-------------------------------")

metricas_Gauss_naive = {
    'media': np.mean(desempenhos_naive_bayes),
    'desvio_padrao': np.std(desempenhos_naive_bayes),
    'maximo': np.max(desempenhos_naive_bayes),
    'minimo': np.min(desempenhos_naive_bayes)
}
print("Gauss (Cov. Igual):")
print(f"Média: {metricas_Gauss_naive['media']}")
print(f"Desvio Padrão: {metricas_Gauss_naive['desvio_padrao']}")
print(f"Valor máximo: {metricas_Gauss_naive['maximo']}")
print(f"Valor mínimo: {metricas_Gauss_naive['minimo']}")
print("-------------------------------")

modelos = (
    "MQO tradicional",
    "Gauss (Tradicional)",
    "Gauss (Cov. Igual)",
    "Gauss (Cov. Agregada)",
    "Gauss regularizado (0,25)",
    "Gauss regularizado (0,50)",
    "Gauss regularizado (0,75)",
    "Gauss Naive"
)

metricas = {
    'Média': 
        (
            metricas_MQO_tradicional['media'],
            metricas_Gauss_tradicional['media'],
            metricas_Gauss_cov_igual['media'],
            metricas_Gauss_cov_agregada['media'],
            metricas_Gauss_regularizado_025['media'],
            metricas_Gauss_regularizado_050['media'],
            metricas_Gauss_regularizado_075['media'],
            metricas_Gauss_naive['media']
        ),
    'Desvio Padrão': 
        (
            metricas_MQO_tradicional['desvio_padrao'],
            metricas_Gauss_tradicional['desvio_padrao'],
            metricas_Gauss_cov_igual['desvio_padrao'],
            metricas_Gauss_cov_agregada['desvio_padrao'],
            metricas_Gauss_regularizado_025['desvio_padrao'],
            metricas_Gauss_regularizado_050['desvio_padrao'],
            metricas_Gauss_regularizado_075['desvio_padrao'],
            metricas_Gauss_naive['desvio_padrao']
        ),
    'Valor máximo': 
        (
            metricas_MQO_tradicional['maximo'],
            metricas_Gauss_tradicional['maximo'],
            metricas_Gauss_cov_igual['maximo'],
            metricas_Gauss_cov_agregada['maximo'],
            metricas_Gauss_regularizado_025['maximo'],
            metricas_Gauss_regularizado_050['maximo'],
            metricas_Gauss_regularizado_075['maximo'],
            metricas_Gauss_naive['maximo']
        ),
    'Valor mínimo': 
        (
            metricas_MQO_tradicional['minimo'],
            metricas_Gauss_tradicional['minimo'],
            metricas_Gauss_cov_igual['minimo'],
            metricas_Gauss_cov_agregada['minimo'],
            metricas_Gauss_regularizado_025['minimo'],
            metricas_Gauss_regularizado_050['minimo'],
            metricas_Gauss_regularizado_075['minimo'],
            metricas_Gauss_naive['minimo']
        ),
}

x = np.arange(len(modelos))  # the label locations
largura = 0.15  # the width of the bars
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
ax.set_ylim(0, 110)


if plot_graphs:
    plt.show()

bp = 1