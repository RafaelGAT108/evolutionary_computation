from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as la
from control.matlab import ss, step
import matplotlib.pyplot as plt
from scipy.integrate import simps
np.seterr(divide='ignore', invalid='ignore')

@dataclass
class GeneticAlgoritmLQR:
    states_size: int = field(init=False)
    input_size: int = field(init=False)
    chromosome_size: int = field(init=False)
    population: np.array = field(init=False)

    best_Q: np.array = field(init=False)
    best_R: np.array = field(init=False)

    generations_size: int
    population_size: int
    A: np.array
    B: np.array
    C: np.array
    D: np.array

    def __post_init__(self):
        self.states_size = self.A.shape[0]
        self.input_size = self.B.shape[1]
        self.chromosome_size = self.states_size + self.input_size ** 2
        self.population = abs(np.random.randn(self.population_size, self.chromosome_size))
        self.best_Q = np.array([])
        self.best_R = np.array([])

    def fit_model(self):
        for gen in range(self.generations_size):
            fitness = np.array([self._fitness_function(chromosome) for chromosome in self.population])

            # parents selection (tournament)
            selected_indices = np.array([np.where(fitness == min(fitness[np.random.choice(self.population_size, size=2)]))[0][0]
                                         for _ in range(self.population_size)])

            parents = self.population[selected_indices]

            # Crossover
            offspring = self._crossover(parents=parents)

            # Mutation
            offspring = np.array([self._mutation(chromosome) for chromosome in offspring])

            self.population = offspring

            print("Generation:", gen, "Error ITAE Min.:", round(np.min(fitness), 3), "Error Sum.:", round(sum(fitness), 3))

        # best chromosome
        best_index = np.argmin(fitness)
        best_chromosome = self.population[best_index]

        # Decode the best chromosome
        self.best_Q = np.diag(best_chromosome[:self.states_size])
        self.best_R = best_chromosome[self.states_size:].reshape((self.input_size, self.input_size))

        return self.best_Q, self.best_R

    def _fitness_function(self, chromosome):

        Q = np.diag(chromosome[:self.states_size])
        R = chromosome[self.states_size:].reshape((self.input_size, self.input_size))

        # Calculate the cost function using LQR
        return self._lqr_cost(Q, R)

    # Function to calculate the cost function of the LQR controller
    def _lqr_cost(self, Q, R):

        try:
            P = la.solve_continuous_are(self.A, self.B, Q, R)
            K = np.linalg.inv(R) @ self.B.T @ P
            sys = ss(self.A - self.B * K, self.B, self.C, self.D)

            t = np.linspace(0, 200, 10000)
            y, t = step(sys, t)  # Response to Unitary Step
            y = np.nan_to_num(y, nan=5000)
            # plt.figure()
            # plt.grid()
            # plt.plot(t, y, 'b')
            # plt.savefig("plotteste.png")

            absolute_error = np.mean(y[:-100]) - y

            # ITAE error
            erro_itae = simps(t * np.abs(absolute_error), t)

            # MSE error
            # erro_mse = np.mean(absolute_error ** 2)

            if np.isnan(erro_itae):
                # return np.inf
                return 5000

            return erro_itae

        except np.linalg.LinAlgError as e:
            # print("Erro durante a solução da equação de Riccati:", e)
            return 5000
            # return float('inf')

    def _mutation(self, chromosome, mutation_rate=0.1):
        mutated_chromosome = chromosome.copy()
        for i in range(len(chromosome)):
            if np.random.rand() < mutation_rate:
                mutated_chromosome[i] += abs(np.random.normal(0, 1))  # gaussian mutation
        return mutated_chromosome

    def _crossover(self, parents, crossover_rate=0.9):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(0, self.chromosome_size, size=self.population_size)
            return np.array([np.concatenate((parents[i][:crossover_point[i]],
                                                  parents[(i + 1) % self.population_size][crossover_point[i]:]))
                                  for i in range(self.population_size)])
        return parents