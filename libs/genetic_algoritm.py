from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as la


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
        self.chromosome_size = self.states_size ** 2 + self.input_size ** 2
        self.population = 2.3 * np.random.randn(self.population_size, self.chromosome_size)
        self.best_Q = np.array([])
        self.best_R = np.array([])

    def fit_model(self):
        for gen in range(self.generations_size):
            fitness = np.array([self._fitness_function(chromosome) for chromosome in self.population])

            # Seleção dos pais (torneio)
            selected_indices = np.random.choice(self.population_size, size=self.population_size, replace=True)
            parents = self.population[selected_indices]

            # Crossover (um ponto de corte)
            crossover_point = np.random.randint(0, self.chromosome_size, size=self.population_size)
            offspring = np.array([np.concatenate((parents[i][:crossover_point[i]],
                                                  parents[(i + 1) % self.population_size][crossover_point[i]:]))
                                  for i in range(self.population_size)])

            # Mutação
            offspring = np.array([self._mutation(chromosome) for chromosome in offspring])

            # Substituição da população antiga pela nova
            population = offspring

            # Mostrar a melhor aptidão a cada geração
            print("Geração:", gen, "Melhor Aptidão:", round(-np.max(fitness), 3))  # Invertemos o sinal para maximizar

        # Encontrar o melhor cromossomo
        best_index = np.argmax(fitness)
        best_chromosome = self.population[best_index]

        # Decodificar o melhor cromossomo
        self.best_Q = best_chromosome[:self.states_size ** 2].reshape((self.states_size, self.states_size))
        self.best_R = best_chromosome[self.states_size ** 2:].reshape((self.input_size, self.input_size))

        return self.best_Q, self.best_R

    def _fitness_function(self, chromosome):
        # Decodificar cromossomo para matrizes de ponderação Q e R
        # TODO: transformar a matriz Q numa matriz diagonal. Assim, ao invés do cromossomo ter 5 values, ele terá 3
        # os dois elementos da diagonal de Q, e o elemento de R

        Q = chromosome[:self.states_size ** 2].reshape((self.states_size, self.states_size))
        R = chromosome[self.states_size ** 2:].reshape((self.input_size, self.input_size))

        # Garantir que Q seja simétrica
        Q = (Q + Q.T) / 2

        # Calcular a função custo usando LQR
        return -self._lqr_cost(Q, R)  # Queremos minimizar a função custo, por isso multiplicamos por -1

    # Função para calcular a função custo do controlador LQR
    def _lqr_cost(self, Q, R):
        # Resolver a equação de Riccati para obter a matriz de ganho L ótima
        try:
                P = la.solve_continuous_are(self.A, self.B, Q, R)
                return np.trace(P)

        except np.linalg.LinAlgError as e:
            # print("Erro durante a solução da equação de Riccati:", e)
            return float('inf')  # Retornar infinito para indicar um custo alto


    def _mutation(self, chromosome, mutation_rate=0.1):
        mutated_chromosome = chromosome.copy()
        for i in range(len(chromosome)):
            if np.random.rand() < mutation_rate:
                mutated_chromosome[i] += np.random.normal(0, 1)  # Mutação gaussiana
        return mutated_chromosome
