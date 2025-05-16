import numpy as np
import matplotlib.pyplot as plt
from config import Config

class Adaline:
    def __init__(self, input_size=1, m=1, η=0.01, ϵ=0.01, epochs=50):
        """
        η: Taxa de aprendizagem (0 < η ≤ 1)
        ϵ: Precisão do modelo (0 < ϵ ≤ 1)
        epochs: Número máximo de épocas para treinamento
        """
        self.__η = η
        self.__input_size = input_size
        self.__m = m
        self.__ϵ = ϵ
        self.__epochs = epochs
        self.__weights = []
        self.__EQMs = []
        # TODO: Descomentar quando implementar o método de treinamento para múltiplas classes
        if m > 1:
            weight_matrix = np.random.random_sample((m, self.__input_size + 1)) - 0.5
            self.__weights.append(weight_matrix)

    def __EQM_multiclass(self, X, y, w):
        """
        Calcula o Erro Quadrático Médio (EQM).
        X: Dados de entrada
        y: Valores reais
        """
        EQM = 0
        n_train_samples = X.shape[0]

        for i in range(n_train_samples):
            # N-ésima amostra de Xtreino
            x_sample = X[i]

            # Forward para obter as previsões para a amostra
            y_pred = w @ x_sample

            # N-ésimo rótulo de Xtreino
            d = y[i]
            d = d.reshape(-1, 1)
            # Calcula o erro quadrático individual (EQI)
            EQI = 0
            for j in range(self.__m):  # Para cada neurônio na camada de saída
                EQI += (d[j] - y_pred[0][j]) ** 2
            if type(EQI) == np.ndarray:
                EQI = np.sum(EQI)

            # Soma o EQI ao EQM
            EQM += EQI

        # Calcula o EQM final
        EQM = EQM / (2 * n_train_samples)

        return EQM

    def __EQM(self, x, y, w):
        """
        Calcula o Erro Quadrático Médio (EQM)
        """
        EQM = 0
        for i in range(len(x)):
            u = w.T @ x[i]
            EQM += (y[i] - u) ** 2
        return EQM / (2 * len(x))

    # TODO: Implementar o método de treinamento para múltiplas classes
    def fit_multiclass(self, X, y):
        X = np.hstack([-1 * np.ones((X.shape[0], 1)), X])  # Adiciona o -1 como primeira coluna (bias)
        epoch = 0
        while True:
                
            w = self.__weights  # Transforma w em uma matriz coluna

            for i in range(len(X)):
                d = y[i]
                u = w @ X[i].reshape(-1, 1)  # Transforma X[i] em uma matriz coluna
                # Calcula o erro
                error = d.reshape(-1, 1) - u
                # Atualiza os pesos
                w += self.__η * (error @ X[i].reshape(-1, 1).T)
            self.__weights = w
            # Calcula o EQM
            EQM = self.__EQM_multiclass(X, y, w)
            print(f"Época {epoch + 1}: EQM = {EQM}")
            self.__EQMs.append(EQM)
            epoch += 1
            # Verifica as condições de parada
            if epoch > 1:
                # Verifica se o módulo da diferença do EQM atual com o anterior é menor que ϵ ou se
                # atingiu o número máximo de épocas
                if abs(self.__EQMs[-1] - self.__EQMs[-2]) <= self.__ϵ or epoch >= self.__epochs:
                    print(f"Convergiu após {epoch} épócas")
                    break
            elif epoch == self.__epochs:
                print(f"Número máximo de épocas atingido: {self.__epochs}")
                break

    def fit(self, X, y):
        if Config.PLOT_REGRESSION_LINE:
            plt.figure(1)
            ax = plt.subplot()
            ax.scatter(X, y, color='blue', label='Dados Normalizados')
        x_axis = np.linspace(-.1, 1.1, 100)
        X = np.hstack([-1 * np.ones((X.shape[0], 1)), X])  # Adiciona o -1 como primeira coluna (bias)
        self.__weights = np.zeros(X.shape[1])  # Inicializa os pesos com zero, incluindo o peso do bias
        epoch = 0
        if Config.PLOT_REGRESSION_LINE:
            l = ax.plot(x_axis, -self.__weights[0] + self.__weights[1] * x_axis, color='red', label='Adaline')
        while True:
            # Calcula a combinação linear
            for i in range(len(X)):
                u = self.__weights.T@X[i]
                # Calcula o erro
                error = y[i] - u
                # Atualiza os pesos
                self.__weights += self.__η * X[i]*error
                if Config.PLOT_REGRESSION_LINE:
                    l[0].remove()
                    l = ax.plot(x_axis, -self.__weights[0] + self.__weights[1] * x_axis, color='red', label='Adaline')
                    plt.pause(.1)
            # Calcula o EQM
            EQM = self.__EQM(X, y, self.__weights)
            print(f"Época {epoch + 1}: EQM = {EQM}")
            self.__EQMs.append(EQM)
            epoch += 1
            # Verifica as condições de parada
            if epoch > 1:
                # Verifica se o módulo da diferença do EQM atual com o anterior é menor que ϵ ou se
                # atingiu o número máximo de épocas
                if abs(self.__EQMs[-1] - self.__EQMs[-2]) <= self.__ϵ or epoch >= self.__epochs:
                    print(f"Convergiu após {epoch} épócas")
                    break
            elif epoch == self.__epochs:
                print(f"Número máximo de épocas atingido: {self.__epochs}")
                break

    def predict_classification(self, X):
        # Adiciona o -1 como primeira coluna (bias)
        X = np.hstack([-1 * np.ones((X.shape[0], 1)), X])
        # Calcula a combinação linear
        linear_output = X @ self.__weights
        # Aplica a função de ativação (step function)
        return np.where(linear_output >= 0.0, 1, -1)

    def predict_multiclass(self, X):
        # Adiciona o -1 como primeira coluna (bias)
        X = np.hstack([-1 * np.ones((X.shape[0], 1)), X])
        return X @ np.array(self.__weights[0]).T
        

    def predict_regression(self, X):
        # Adiciona o -1 como primeira coluna (bias)
        X = np.hstack([-1 * np.ones((X.shape[0], 1)), X])
        # Retorna a combinação linear
        return X @ self.__weights

    def plot_EQMs(self, title):
        # Plota o EQM ao longo das épocas
        plt.plot(range(1, len(self.__EQMs) + 1), self.__EQMs, marker='o')
        plt.xlabel('epochs')
        plt.ylabel('EQM')
        plt.title(title)