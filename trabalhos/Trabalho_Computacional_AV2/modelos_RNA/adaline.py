import numpy as np
import matplotlib.pyplot as plt

class Adaline:
    def __init__(self, η=0.01, ϵ=0.01, epochs=50):
        """
        η: Taxa de aprendizagem (0 < η ≤ 1)
        ϵ: Precisão do modelo (0 < ϵ ≤ 1)
        epochs: Número máximo de épocas para treinamento
        """
        self.__η = η
        self.__ϵ = ϵ
        self.__epochs = epochs
        self.__weights = []
        self.__EQMs = []

    def __EQM(self, x, y, w):
        """
        Calcula o Erro Quadrático Médio (EQM)
        """
        EQM = 0
        for i in range(len(x)):
            u = w.T @ x[i]
            EQM += (y[i] - u) ** 2
        return EQM / (2 * len(x))

    def fit(self, X, y):
        X = np.hstack([-1 * np.ones((X.shape[0], 1)), X])  # Adiciona o -1 como primeira coluna (bias)
        self.__weights = np.zeros(X.shape[1])  # Inicializa os pesos com zero, incluindo o peso do bias
        epoch = 0
        while True:
            # Calcula a combinação linear
            linear_combination = X@self.__weights
            # Calcula os erros
            errors = y - linear_combination
            # Atualiza os pesos
            self.__weights += self.__η * X.T@errors
            # Calcula o EQM
            EQM = self.__EQM(X, y, self.__weights)
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

    def predict_regression(self, X):
        # Adiciona o -1 como primeira coluna (bias)
        X = np.hstack([-1 * np.ones((X.shape[0], 1)), X])
        # Retorna a combinação linear
        return X @ self.__weights

    def plot_EQMs(self):
        # Plota o EQM ao longo das épocas
        plt.plot(range(1, len(self.__EQMs) + 1), self.__EQMs, marker='o')
        plt.xlabel('epochs')
        plt.ylabel('EQM')
        plt.title('Adaline - EQM over epochs')