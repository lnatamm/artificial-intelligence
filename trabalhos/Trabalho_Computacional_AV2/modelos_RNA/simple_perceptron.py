import numpy as np

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

    def fit(self, X, y):
        X = np.hstack([-1 * np.ones((X.shape[0], 1)), X])  # Adiciona o -1 como primeira coluna (bias)
        self.__weights = np.zeros(X.shape[1])  # Inicializa os pesos com zero, incluindo o peso do bias
        epoch = 0
        while True:
            # Calcula a combinação linear
            linear_combination = X@self.__weights
            # Aplica a função de ativação (step function)
            output = np.where(linear_combination >= 0.0, 1, -1)
            # Calcula os erros
            errors = y - output
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