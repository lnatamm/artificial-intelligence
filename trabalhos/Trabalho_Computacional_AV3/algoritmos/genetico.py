import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgorithm:

    def __init__(self, f, N, p, min, max, n_bits=5, tournament_size=1/3, mutation_probability=0.02, crossover_probability=0.85, max_generations=10):
        self.f = f
        self.N = N
        self.p = p
        self.min = min
        self.max = max
        self.n_bits = n_bits
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.max_generations = max_generations
        self.population = np.random.randint(0, 2, size=(self.N, self.n_bits * self.p))
        self.best_individual = None

    def plot_function(self):
        figure = plt.figure()
        ax = figure.add_subplot(projection='3d')
        x_axis = np.linspace(self.min, self.max, 100)
        mesh_x, mesh_y = np.meshgrid(x_axis, x_axis)
        ax.plot_surface(mesh_x, mesh_y, self.f(mesh_x, mesh_y), cmap='viridis')

    def plot_population(self):
        # Cria a figura e superfície apenas uma vez
        if not hasattr(self, 'fig'):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection='3d')
            x_axis = np.linspace(self.min, self.max, 100)
            mesh_x, mesh_y = np.meshgrid(x_axis, x_axis)
            self.ax.plot_surface(mesh_x, mesh_y, self.f(mesh_x, mesh_y), cmap='viridis', alpha=0.5)
            self.population_points = []
        # Remove todos os pontos anteriores
        for point in getattr(self, 'population_points', []):
            point.remove()
        self.population_points = []
        # Plota a população atual
        for individual in self.population:
            decoded_individual = [self._decode(individual[i:i+self.n_bits]) for i in range(0, len(individual), self.n_bits)]
            point = self.ax.scatter(decoded_individual[0], decoded_individual[1], self.f(*decoded_individual), color='red')
            self.population_points.append(point)
        plt.draw()
        plt.pause(0.5)

    def _decode(self, individual):
        # Convert bit array to integer (most significant bit first)
        value = 0
        for i in range(self.n_bits):
            value += individual[i] * (2 ** (self.n_bits - i - 1))
        # Scale to [min, max]
        return self.min + (self.max - self.min) * value / ((2 ** self.n_bits) - 1)

    def _apply_fitness(self, splitted_population):
        fitness = []
        for individual in splitted_population:
            decoded_individual = [self._decode(feature) for feature in individual]
            fitness.append(self.f(*decoded_individual))
        return np.array(fitness)

    def _select(self, fitness):
        selected = []
        for i in range(self.N):
            tournament = np.random.permutation(self.N)[:int(self.tournament_size * self.N)]
            winner = tournament[np.argmax([fitness[j] for j in tournament])]
            selected.append(self.population[winner])
        return np.array(selected)

    def _crossover(self, selected_individuals):
        for i in range(len(selected_individuals) - 1):
            father = selected_individuals[i]
            mother = selected_individuals[i+1]
            if np.random.rand() < self.crossover_probability:
                mask = np.random.randint(0, 2, self.n_bits * self.p)
                first_child = np.copy(father)
                second_child = np.copy(mother)
                for j in range(len(mask)):
                    if mask[j] == 1:
                        first_child[j] = mother[j]
                        second_child[j] = father[j]
                selected_individuals[i] = first_child
                selected_individuals[i+1] = second_child
        return selected_individuals

    def _mutate(self, selected_individuals):
        for individual in selected_individuals:
            for i in range(len(individual)):
                if np.random.rand() < self.mutation_probability:
                    individual[i] = 1 - individual[i]
        return selected_individuals

    def start(self):
        generation = 0
        while generation < self.max_generations:
            splitted_population = np.array([np.array_split(individual, self.p) for individual in self.population])
            fitness = self._apply_fitness(splitted_population)
            decoded_best = [self._decode(feature) for feature in self.best_individual] if self.best_individual is not None else None
            if generation == 0 or np.max(fitness) > self.f(*decoded_best):
                self.best_individual = splitted_population[np.argmax(fitness)]
                print(f"Generation {generation}: Best fitness = {np.max(fitness)}, Best individual = {self.best_individual}, Decoded = {decoded_best}")
            else:
                print(f"Generation {generation}: No improvement")
            
            selected_individuals = self._select(fitness)
            selected_individuals = self._crossover(selected_individuals)
            selected_individuals = self._mutate(selected_individuals)
            self.population = np.array(selected_individuals)
            self.plot_population()
            generation += 1
        return [self._decode(feature) for feature in self.best_individual]
    