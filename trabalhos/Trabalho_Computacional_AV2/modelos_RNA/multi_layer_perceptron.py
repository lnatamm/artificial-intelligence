import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, q, m, η=0.0001, ϵ=0.0001, tolleration=2, epochs=50):
        """
        input_size: Number of features in the input data
        L: Número de camadas ocultas
        q: Lista com a quantidade de neurônios em cada camada oculta
        m: Número de neurônios na camada de saída
        η: Taxa de aprendizagem (0 < η ≤ 1)
        ϵ: Precisão do modelo (0 < ϵ ≤ 1)
        tolleration: Número máximo de vezes que o modelo pode piorar antes de parar
        """
        self.__input_size = input_size
        self.__q = q
        self.__m = m
        self.__η = η
        self.__ϵ = ϵ
        self.__tolleration = tolleration
        self.__epochs = epochs

        # Initialize weights and biases for each layer
        self.__weights = []
        # Inicializa os EQMs
        self.__EQMs = []
        # Histórico de pesos
        self.__weights_history = []

        # Lista com o tamanho de cada camada: entrada + ocultas + saída
        layer_sizes = [self.__input_size] + self.__q + [self.__m]

        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            # Inicializa pesos com valores entre -0.5 e 0.5 e inclui o bias (+1)
            weight_matrix = np.random.random_sample((n_in + 1, n_out)) - 0.5
            self.__weights.append(weight_matrix)
        # Salva os pesos iniciais
        self.__weights_history.append(self.__weights.copy())

    # Função de ativação sigmoide logística
    def __sigmoid_logistic(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivada da função de ativação sigmoide logística
    def __sigmoid_logistic_derivative(self, x):
        return x * (1 - x)

    # Forward usado para tarefa de classificação
    def forward_classification(self, x_sample):
        """
        Modelo de previsão para classificação.
        x_sample: X de teste
        """
        self.__layer_inputs = []
        self.__layer_outputs = []

        for j, W in enumerate(self.__weights):
            # Primeira camada: Camada de entrada
            if j == 0:
                # Combinação linear
                layer_input = W.T @ x_sample
                layer_input = layer_input.reshape(-1, 1)
                # Aplicação da função de ativação
                layer_output = self.__sigmoid_logistic(layer_input)
                self.__layer_inputs.append(layer_input)
                self.__layer_outputs.append(layer_output)
            # Camadas ocultas e de saída
            else:
                # Adição do bias à saída da camada anterior
                y_bias = np.vstack([-1 * np.ones((1, self.__layer_outputs[-1].shape[1])), self.__layer_outputs[-1]])
                # Combinação linear
                layer_input = W.T @ y_bias
                # Aplicação da função de ativação
                layer_output = self.__sigmoid_logistic(layer_input)
                self.__layer_inputs.append(layer_input)
                self.__layer_outputs.append(layer_output)
        # Retorna a saída da última camada
        return self.__layer_outputs[-1].T

    def forward_regression(self, x_sample):
        """
        Modelo de previsão para regressão.
        x_sample: X de teste
        """
        self.__layer_inputs = []
        self.__layer_outputs = []

        for j, W in enumerate(self.__weights):
            # Primeira camada: Camada de entrada
            if j == 0:
                # Combinação linear
                layer_input = W.T @ x_sample
                layer_input = layer_input.reshape(-1, 1)
                # Sem função de ativação, pois a tarefa é de regressão
                layer_output = layer_input
                self.__layer_inputs.append(layer_input)
                self.__layer_outputs.append(layer_output)
            # Camadas ocultas e de saída
            else:
                # Adição do bias à saída da camada anterior
                y_bias = np.vstack([-1 * np.ones((1, self.__layer_outputs[-1].shape[1])), self.__layer_outputs[-1]])
                # Combinação linear
                layer_input = W.T @ y_bias
                # Sem função de ativação, pois a tarefa é de regressão
                layer_output = layer_input
                self.__layer_inputs.append(layer_input)
                self.__layer_outputs.append(layer_output)

        return self.__layer_outputs[-1].T
    
    def __backward_regression(self, x_sample, d):
        """
        Algoritmo de backpropagation para atualizar pesos e biases.
        x_sample: Amostra de entrada
        d: Desejado
        """
        j = len(self.__weights) - 1
        # Inicializa a lista de gradientes
        δ = []
        for i in range(len(self.__weights)):
            δ.append([])
        while j >= 0:
            # Camada de saída
            if j + 1 == len(self.__weights):
                # Gradiente da função de ativação
                δ[j] = self.__layer_inputs[j] * (d - self.__layer_outputs[j])
                # adiciona o bias no y
                y_bias = np.vstack([-1 * np.ones((1, self.__layer_outputs[j-1].shape[1])), self.__layer_outputs[j-1]])
                # Atualiza os pesos
                # Aplicamos o .T para que a soma de matrizes funcione
                # Já que self.__weights[j] tem dimensões input_size x m e δ[j] @ y_bias.T tem dimensões m x input_size
                self.__weights[j] += (self.__η * (δ[j] @ y_bias.T)).T
            # Camada de entrada
            elif j == 0:
                # W[j+1] recebe a matriz W[j+1] sem a coluna que multiplica o bias
                W_without_bias = self.__weights[j+1][1:, :]
                # Gradiente da função de ativação
                δ[j] = self.__layer_inputs[j] * W_without_bias @ δ[j+1]
                # Atualiza os pesos
                # Aplicamos o .T para que a soma de matrizes funcione
                # Já que self.__weights[j] tem dimensões input_size x m e δ[j] @ y_bias.T tem dimensões m x input_size
                self.__weights[j] += self.__η * (δ[j] @ x_sample.T).T
            else:
                # W[j+1] recebe a matriz W[j+1] sem a coluna que multiplica o bias
                W_without_bias = self.__weights[j+1][1:, :]
                # Gradiente da função de ativação
                δ[j] = self.__layer_inputs[j] * W_without_bias @ δ[j+1]
                # adiciona o bias no y
                y_bias = np.vstack([-1 * np.ones((1, self.__layer_outputs[j-1].shape[1])), self.__layer_outputs[j-1]])
                # Atualiza os pesos
                # Aplicamos o .T para que a soma de matrizes funcione
                # Já que self.__weights[j] tem dimensões input_size x m e δ[j] @ y_bias.T tem dimensões m x input_size
                self.__weights[j] += self.__η * (δ[j] @ y_bias.T).T
            j -= 1
        # Salva os pesos após a atualização
        self.__weights_history.append(self.__weights.copy())

    def __backward_classification(self, x_sample, d):
        """
        Algoritmo de backpropagation para atualizar pesos e biases.
        x_sample: Amostra de entrada
        d: Desejado
        """
        j = len(self.__weights) - 1
        # Inicializa a lista de gradientes
        δ = []
        for i in range(len(self.__weights)):
            δ.append([])
        while j >= 0:
            # Camada de saída
            if j + 1 == len(self.__weights):
                # Gradiente da função de ativação
                δ[j] = self.__sigmoid_logistic_derivative(self.__layer_inputs[j]) * (d - self.__layer_outputs[j])
                # adiciona o bias no y
                y_bias = np.vstack([-1 * np.ones((1, self.__layer_outputs[j-1].shape[1])), self.__layer_outputs[j-1]])
                # Atualiza os pesos
                # Aplicamos o .T para que a soma de matrizes funcione
                # Já que self.__weights[j] tem dimensões input_size x m e δ[j] @ y_bias.T tem dimensões m x input_size
                self.__weights[j] += (self.__η * (δ[j] @ y_bias.T)).T
            # Camada de entrada
            elif j == 0:
                # W[j+1] recebe a matriz W[j+1] sem a coluna que multiplica o bias
                W_without_bias = self.__weights[j+1][1:, :]
                # Gradiente da função de ativação
                δ[j] = self.__sigmoid_logistic_derivative(self.__layer_inputs[j]) * W_without_bias @ δ[j+1]
                # Atualiza os pesos
                # Aplicamos o .T para que a soma de matrizes funcione
                # Já que self.__weights[j] tem dimensões input_size x m e δ[j] @ x_sample.T tem dimensões m x input_size
                self.__weights[j] += self.__η * (δ[j] @ x_sample.T).T
            else:
                # W[j+1] recebe a matriz W[j+1] sem a coluna que multiplica o bias
                W_without_bias = self.__weights[j+1][1:, :]
                # Gradiente da função de ativação
                δ[j] = self.__sigmoid_logistic_derivative(self.__layer_inputs[j]) * W_without_bias @ δ[j+1]
                # adiciona o bias no y
                y_bias = np.vstack([-1 * np.ones((1, self.__layer_outputs[j-1].shape[1])), self.__layer_outputs[j-1]])
                # Atualiza os pesos
                # Aplicamos o .T para que a soma de matrizes funcione
                # Já que self.__weights[j] tem dimensões input_size x m e δ[j] @ y_bias.T tem dimensões m x input_size 
                self.__weights[j] += self.__η * (δ[j] @ y_bias.T).T
            j -= 1
        # Salva os pesos após a atualização
        self.__weights_history.append(self.__weights.copy())

    def __EQM_classification(self, X, y):
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
            y_pred = self.forward_classification(x_sample)

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

    def __EQM_regression(self, X, y):
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
            x_sample.shape = (x_sample.shape[0], 1)  # Reshape para garantir que seja uma matriz 2D

            # Forward para obter as previsões para a amostra
            y_pred = self.forward_regression(x_sample)

            # N-ésimo rótulo de Xtreino
            d = y[i]

            # Calcula o erro quadrático individual (EQI)
            EQI = 0
            for j in range(self.__m):  # Para cada neurônio na camada de saída
                EQI += (d[j] - y_pred[0][j]) ** 2

            # Soma o EQI ao EQM
            EQM += EQI

        # Calcula o EQM final
        EQM = EQM / (2 * n_train_samples)

        return EQM
    
    def __least_eqm(self, EQMs):
        """
        Retorna o índice do menor EQM
        """
        min_eqm = min(EQMs)
        return EQMs.index(min_eqm)

    def train_classification(self, X, y, X_validation, y_validation):
        """
        Treinamento do modelo para classificação.
        X: Dados de entrada
        y: Valores reais
        X_validation: Dados de validação
        y_validation: Valores reais de validação
        epochs: Número de épocas para treinamento (Limitado pelo número máximo de épocas definido na inicialização)
        """
        # Adiciona o -1 como primeira coluna (bias)
        X = np.hstack([-1 * np.ones((X.shape[0], 1)), X])
        X_validation = np.hstack([-1 * np.ones((X_validation.shape[0], 1)), X_validation])
        EQM = 1
        # Salva os EQMs de validação
        EQMs_validation = []
        # Contador de tolerância
        tolleration_count = 0
        while True:
            for epoch in range(self.__epochs):
                for i in range(X.shape[0]):  # Passa cada amostra individualmente
                    x_sample = X[i]
                    x_sample = x_sample.reshape(len(x_sample), 1)
                    y_sample = y[i]
                    y_sample = y_sample.reshape(-1, 1)
                    output = self.forward_classification(x_sample)
                    # Atualiza os pesos
                    self.__backward_classification(x_sample, y_sample)
                # Calcula o EQM
                EQM = self.__EQM_classification(X, y)
                print(f"Epoch {epoch}, Loss: {EQM:.4f}")
                self.__EQMs.append(EQM)
                
                # Verifica se o modelo está piorando
                if epoch > 1:
                    # Faz a validação a cada 5 épocas
                    if epoch % 5 == 0:
                        print("Validando o modelo...")
                        EQM_validation = self.__EQM_classification(X_validation, y_validation)
                        print(f"Epoch {epoch}, EQM Validation: {EQM_validation:.4f}")
                        EQMs_validation.append(EQM_validation)
                        if len(EQMs_validation) > 1:
                            if EQMs_validation[-1] > EQMs_validation[-2]:
                                tolleration_count += 1
                                print(f"Modelo piorou. Contagem de tolerância: {tolleration_count}/{self.__tolleration}")
                                if tolleration_count >= self.__tolleration:
                                    print(f"Early stopping ativado após {epoch} épocas devido à piora contínua.")
                                    self.__weights = self.__weights_history[self.__least_eqm(self.__EQMs)]
                                    return
                            else:
                                tolleration_count = 0  # Reseta a contagem se o modelo melhorar
                    # Verifica se EQM atual é menor que a precisão do modelo
                    if self.__EQMs[-1] < self.__ϵ:
                        print(f"Convergiu após {epoch} épocas")
                        return
                
            print(f"Número de épocas atingido: {self.__epochs}")
            return
    
    def train_regression(self, X, y, X_validation, y_validation):
        # Adiciona o -1 como primeira coluna (bias)
        X = np.hstack([-1 * np.ones((X.shape[0], 1)), X])
        X_validation = np.hstack([-1 * np.ones((X_validation.shape[0], 1)), X_validation])
        EQM = 1
        # Salva os EQMs de validação
        EQMs_validation = []
        # Contador de tolerância
        tolleration_count = 0
        while True:
            for epoch in range(self.__epochs):
                for i in range(X.shape[0]):  # Passa cada amostra individualmente
                    x_sample = X[i]
                    x_sample.shape = (x_sample.shape[0], 1)  # Reshape para garantir o shape correto
                    y_sample = y[i]
                    y_sample.shape = (y_sample.shape[0], 1)  # Reshape para garantir o shape correto
                    output = self.forward_regression(x_sample)
                    # Atualiza os pesos
                    self.__backward_regression(x_sample, y_sample)
                # Calcula o EQM
                EQM = self.__EQM_regression(X, y)
                print(f"Epoch {epoch}, Loss: {EQM:.4f}")
                self.__EQMs.append(EQM)
                
                # Verifica se o modelo está piorando
                if epoch > 1:
                    # Faz a validação a cada 5 épocas
                    if epoch % 5 == 0:
                        print("Validando o modelo...")
                        EQM_validation = self.__EQM_regression(X_validation, y_validation)
                        print(f"Epoch {epoch}, EQM Validation: {EQM_validation:.4f}")
                        EQMs_validation.append(EQM_validation)
                        if len(EQMs_validation) > 1:
                            if EQMs_validation[-1] > EQMs_validation[-2]:
                                tolleration_count += 1
                                print(f"Modelo piorou. Contagem de tolerância: {tolleration_count}/{self.__tolleration}")
                                if tolleration_count >= self.__tolleration:
                                    print(f"Early stopping ativado após {epoch} épocas devido à piora contínua.")
                                    self.__weights = self.__weights_history[self.__least_eqm(self.__EQMs)]
                                    return
                            else:
                                tolleration_count = 0  # Reseta a contagem se o modelo melhorar
                    # Verifica se EQM atual é menor que a precisão do modelo
                    if self.__EQMs[-1] < self.__ϵ:
                        print(f"Convergiu após {epoch} épocas")
                        return
            print(f"Número máximo de épocas atingido: {self.__epochs}")
            return
    def predict_classification(self, X):
        return self.forward_classification(X)

    def predict_regression(self, X):
        # Retorna a combinação linear
        return self.forward_regression(X)
    
    def plot_EQMs(self):
        """
        Plota os EQMs
        """
        plt.plot(range(1, len(self.__EQMs) + 1), self.__EQMs, marker='o')
        plt.title('EQM')
        plt.xlabel('Épocas')
        plt.ylabel('EQM')