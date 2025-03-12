import numpy as np
import matplotlib.pyplot as plt

# Tarefa de Regressão

# Configuração padrão dos plots
def get_plot_configuration(file, n_figure):
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
    plot.set_title("Atividade Enzimática")

    return plot

# 1.
atividade_enzimatica = np.loadtxt("atividade_enzimatica.csv", delimiter=",")

plot_1 = get_plot_configuration(atividade_enzimatica, 1)

# 2.
X = atividade_enzimatica[:,:2]

y = atividade_enzimatica[:,-1]

# 3.
# Constantes
n_linspace = 40
x_axis = np.linspace(-4, 6, n_linspace)
y_axis = np.linspace(-9, 9, n_linspace)
X3d, Y3d = np.meshgrid(x_axis, y_axis)

# MQO Tradicional
plot_MQO_tradicional = get_plot_configuration(atividade_enzimatica, 2)

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

# lambdas_MQO_regularizado = [0, 0.25, 0.5, 0.75, 1]

# for i, lambda_MQO_regularizado in enumerate(lambdas_MQO_regularizado):
#     plot_MQO_regularizado = get_plot_configuration(atividade_enzimatica, (4+i))
#     B_MQO_regularizado = np.linalg.pinv(X_MQO_tradicional.T@X_MQO_tradicional + lambda_MQO_regularizado[i]@np.identity)@X_MQO_tradicional.T@y
#     Z_MQO_regularizado = B_MQO_regularizado[0] + B_MQO_regularizado[1]*X3d + B_MQO_regularizado[2]*Y3d

# Média
plot_media = get_plot_configuration(atividade_enzimatica, 3)
media = np.mean(y)

B_media = [
    media,
    0,
    0
]

Z_media = B_media[0] + B_media[1]*X3d + B_media[2]*Y3d

plot_media.plot_surface(X3d, Y3d, Z_media, cmap='gray')

plt.show()