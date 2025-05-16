import numpy as np
import matplotlib.pyplot as plt

class SimplePerceptron:
    def __init__(self, η=0.01, epochs=50):
        """
        η: Taxa de aprendizagem (0 < η ≤ 1)
        ϵ: Precisão do modelo (0 < ϵ ≤ 1)
        epochs: Número máximo de épocas para treinamento
        """
        self.__η = η
        self.__epochs = epochs
        self.__weights = []
        self.__errors = []

    def fit(self, X, y):
        X = np.hstack([-1 * np.ones((X.shape[0], 1)), X])  # Adiciona o -1 como primeira coluna (bias)
        self.__weights = np.zeros(X.shape[1])  # Inicializa os pesos com zero, incluindo o peso do bias
        epoch = 0
        while True:
            # Calcula a combinação linear
            errors = []
            for i in range(len(X)):
                u = self.__weights.T @ X[i]
                # Aplica a função de ativação (step function)
                output = 1 if u >= 0.0 else -1
                # Calcula o erro
                error = y[i] - output
                errors.append(error)
                # Atualiza os pesos
                self.__weights += self.__η * X[i] * error
            self.__errors.append(np.sum(np.where(errors != 0, 1, errors)) / len(X))
            if np.all(errors == 0):
                print(f"Convergiu após {epoch} épócas")
                break
            # Atualiza os pesos
            self.__weights += self.__η * X.T@errors
            epoch += 1
            if epoch >= self.__epochs:
                print(f"Número máximo de épocas atingido: {self.__epochs}")
                break

    def predict(self, X):
        # Adiciona o -1 como primeira coluna (bias)
        X = np.hstack([-1 * np.ones((X.shape[0], 1)), X])
        # Calcula a combinação linear
        linear_output = X @ self.__weights
        # Aplica a função de ativação (step function)
        return np.where(linear_output >= 0.0, 1, -1)

    def plot_errors(self, title='Erros'):
        plt.plot(self.__errors)
        plt.title(title)
        plt.xlabel('Épocas')
        plt.ylabel('Erro')
        plt.ylim(0, 1.1)