from dataclasses import dataclass, field
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')


@dataclass
class GeneticAlgorithmBinary:
    cost_function: callable
    generations_size: int
    population_size: int
    selection_method: str
    elitism: bool
    variables: list = list
    chromosome_size: int = 22
    best_x1: Union[float, None] = None
    best_x2: Union[float, None] = None
    population: np.array = field(init=False)
    num_elites: int = field(init=False)
    best_fitness: list = list
    mean_fitness: list = list
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1

    def __post_init__(self):
        self.population = np.random.randint(0, 2, size=(self.population_size, self.chromosome_size))
        self.num_elites = int(0.1 * self.population_size)
        self.best_fitness = []
        self.mean_fitness = []

    def fit_model(self):
        for gen in range(self.generations_size):
            fitness = np.array([self._fitness_function(chromosome) for chromosome in self.population])
            # if len(self.mean_fitness) < 30:
            self.best_fitness.append(round(np.min(-fitness), 3))
            self.mean_fitness.append(round(np.mean(-fitness), 3))

            if self.elitism:
                elite_indices = np.argsort(fitness)[-self.num_elites:]
                elite_population = self.population[elite_indices]

            if self.selection_method == "tournament":
                # parents selection (tournament)
                if self.elitism:
                    selected_index = np.array([np.where(fitness == max(fitness[np.random.choice(self.population_size, size=2)]))[0][0]
                                               for _ in range(self.population_size - self.num_elites)])
                else:
                    selected_index = np.array([np.where(fitness == max(fitness[np.random.choice(self.population_size, size=2)]))[0][0]
                                               for _ in range(self.population_size)])

            elif self.selection_method == "roullete":
                selection_probability = abs(fitness) / np.sum(abs(fitness))
                if self.elitism:
                    selected_index = np.random.choice(self.population_size, size=self.population_size - self.num_elites, p=selection_probability)
                else:
                    selected_index = np.random.choice(self.population_size, size=self.population_size, p=selection_probability)

                # parents = population[select_index]

            else:
                raise Exception("Selection_method invalid !")
            parents = self.population[selected_index]

            # Crossover
            offspring = self._crossover(parents=parents)

            # Mutation
            offspring = np.array([self._mutation(chromosome) for chromosome in offspring])

            if self.elitism:
                self.population = np.vstack((elite_population, offspring))
            else:
                self.population = offspring
            # print("Generation:", gen + 1, "Min.:", round(np.min(-fitness), 4))

        # best chromosome
        best_index = np.argmax(fitness)

        # print("-" * 100)
        # print("\nConjunto de Soluções Encontradas:")

        # best_index = np.argmax(fitness)
        best_chromosome = self.population[best_index]
        variables = self._binary_to_decimal(best_chromosome)

        # print(f"X1: {variables[0]}, X2: {variables[1]}")
        # print(f"Mínimo Encontrado: {round(np.min(-fitness), 4)}")

        return variables[0], variables[1], round(np.min(-fitness), 3), round(np.mean(-fitness), 3)

    def _fitness_function(self, chromosome):

        x1, x2 = self._binary_to_decimal(chromosome)

        return - self.cost_function(x1, x2)  # multiply per -1 to minimize

    def _binary_to_decimal(self, chromosome):

        x1 = int(''.join(chromosome[1:11].astype('str').tolist()), 2) / 100
        if chromosome[0] == 1:
            x1 = -x1

        x2 = int(''.join(chromosome[12:21].astype('str').tolist()), 2) / 100
        if chromosome[11] == 1:
            x2 = -x2

        return x1, x2

    def _mutation(self, chromosome):
        mutated_chromosome = chromosome.copy()
        for i in range(len(chromosome)):
            if np.random.rand() < self.mutation_rate:
                mutated_chromosome[i] = 0 if mutated_chromosome[i] == 1 else 1
        return mutated_chromosome

    def _crossover(self, parents):
        cross_population = []
        if self.elitism:
            value = self.population_size - self.num_elites
        else:
            value = self.population_size

        for i in range(value):
            if np.random.rand() < self.crossover_rate:
                crossover_point_x1 = np.random.randint(0, self.chromosome_size // 2)
                crossover_point_x2 = 1 + self.chromosome_size // 2 + np.random.randint(1, self.chromosome_size // 2)

                if self.elitism:
                    cross_population.append(np.block([parents[i][:crossover_point_x1],
                                                      parents[(i + 1) % (self.population_size - self.num_elites)][crossover_point_x1:11],
                                                      parents[i][11:crossover_point_x2],
                                                      parents[(i + 1) % (self.population_size - self.num_elites)][crossover_point_x2: 22]]))

                else:
                    cross_population.append(np.block([parents[i][:crossover_point_x1],
                                                      parents[(i + 1) % (self.population_size)][crossover_point_x1:11],
                                                      parents[i][11:crossover_point_x2],
                                                      parents[(i + 1) % (self.population_size)][crossover_point_x2: 22]]))


            else:
                cross_population.append(parents[i])

        return cross_population

    def roullete(self, population, fitness):
        selection_probability = fitness / np.sum(fitness)
        select_index = np.random.choice(len(population), size=len(population), p=selection_probability)
        parents = population[select_index]
        return parents


if __name__ == "__main__":

    def shubert_cost(x1, x2):
        sum1 = 0
        sum2 = 0
        for i in range(1, 6):
            new1 = i * np.cos((i + 1) * x1 + i)
            new2 = i * np.cos((i + 1) * x2 + i)
            sum1 += new1
            sum2 += new2

        return sum1 * sum2


    def levy_cost(x1, x2):
        term1 = (np.sin(3 * np.pi * x1)) ** 2
        term2 = (x1 - 1) ** 2 * (1 + (np.sin(3 * np.pi * x2)) ** 2)
        term3 = (x2 - 1) ** 2 * (1 + (np.sin(2 * np.pi * x2)) ** 2)

        return term1 + term2 + term3


    # mutation_rate = [.1, .05, .2]
    mutation_rate = [.05]
    # crossover_rate = [.9, .8, .7]
    crossover_rate = [.7]
    # population_size = [40 * 22, 20 * 22]
    population_size = [880]
    # generations_size = [30, 60]
    generations_size = [30]
    # selection_method = ["roullete", "tournament"]
    selection_method = ["tournament"]
    # elistism = [True, False]
    elistism = [True]

    executions = []
    best_fitness = []
    boxplot_data = np.empty((30, 0))
    boxplot_data_mean = np.empty((30, 0))
    j = 1

    for g_value in generations_size:
        for method in selection_method:
            for elit in elistism:
                for crossover_value in crossover_rate:
                    for mutation_value in mutation_rate:
                        for p_value in population_size:

                            best_fitness = []
                            mean_fitness = []
                            for i in range(30):
                                GA = GeneticAlgorithmBinary(
                                    population_size=p_value,
                                    generations_size=g_value,
                                    cost_function=shubert_cost,
                                    selection_method=method,
                                    mutation_rate=mutation_value,
                                    crossover_rate=crossover_value,
                                    elitism=elit
                                )

                                x1, x2, best_f, mean_f = GA.fit_model()
                                best_fitness.append(best_f)
                                mean_fitness.append(mean_f)
                                print(f"(P_{p_value} G_{g_value} C_{crossover_value} M_{mutation_value}, S_{method}), E({elit})")
                                print(f"Setup: C{j} Exec.: {i} Fitness Best: {best_f} Fitness Mean: {mean_f}\n")

                            # print(f"(P_{p_value} G_{g_value} C_{crossover_value} M_{mutation_value}, S_{method})")
                            # print("Fitness Mean: ", round(np.mean(GA.best_fitness), 3))
                            # print("Fitness Median: ", np.median(GA.best_fitness))
                            # print("Fitness Min.: ", np.min(GA.best_fitness))
                            # print("Fitness Max.: ", np.max(GA.best_fitness))
                            boxplot_data = np.hstack((boxplot_data, np.array(best_fitness).reshape(-1, 1)))
                            boxplot_data_mean = np.hstack((boxplot_data_mean, np.array(mean_fitness).reshape(-1, 1)))
                            executions.append({
                                "Index": f"C{j}",
                                "Population_size": p_value,
                                "generation_size": g_value,
                                "Crossover_rate": crossover_value,
                                "Mutation_rate": mutation_value,
                                "Selection Method": method,
                                "Elitism": elit,
                                "Fitness_Mean": round(np.mean(best_fitness), 3),
                                "Fitness_Median: ": np.median(best_fitness),
                                "Fitness_Min.": np.min(best_fitness),
                                "Fitness_Max.: ": np.max(best_fitness),
                                # "X1": x1,
                                # "X2": x2
                            })
                            j += 1
                            t = [i for i in range(30)]
                            plt.figure(figsize=(10, 5))
                            plt.grid()
                            plt.plot(t, best_fitness, 'b')
                            plt.title(f"Best Features (P_{p_value} G_{g_value} C_{crossover_value} M_{mutation_value}, S_{method}), E({elit})")
                            plt.savefig(f"images2/shubert_cost/Best Features (P_{p_value} G_{g_value} C_{crossover_value} M_{mutation_value},S_{method}, E({elit})).png")

                            plt.figure(figsize=(10, 5))
                            plt.grid()
                            plt.plot(t, mean_fitness, 'b')
                            plt.title(f"Mean Features (P_{p_value} G_{g_value} C_{crossover_value} M_{mutation_value}) S_{method}), E({elit})")
                            plt.savefig(f"images2/shubert_cost/Mean Features (P_{p_value} G_{g_value} C_{crossover_value} M_{mutation_value}, S_{method}, E({elit})).png")

    plt.figure(figsize=(10, 5))

    plt.title(f"Mean Features")

    df2 = pd.DataFrame(boxplot_data_mean, columns=['P220', 'P440', "P880"])
    boxplot2 = df2.boxplot()
    fig2 = boxplot2.get_figure()
    # fig.title("Mean Values Features")
    fig2.savefig('boxplot_mean_levy_cost_C8_final.png')

    plt.figure(figsize=(10, 5))
    # boxplot_data = np.array(boxplot_data)

    plt.title(f"Best Features")
    df = pd.DataFrame(boxplot_data, columns=['P220', 'P440', "P880"])
    boxplot = df.boxplot()
    fig = boxplot.get_figure()
    #fig.title("Best Values Features")
    fig.savefig('boxplot_best_levy_cost_C8_final.png')

    executions = sorted(executions, key=lambda x: x["Fitness_Min."])
    executions = pd.DataFrame(executions)
    # executions.to_excel("results_shubert_cost.xlsx", index=False)
