from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class Particle:
    position: np.ndarray
    fitness: float | None = None


@dataclass
class ClonalG:
    population: list[Particle] = field(init=False)
    selection_rate: float
    cloning_rate: float
    mutation_range: float
    population_size: int
    cost_function: callable
    interations_size: int

    # population

    def __post_init__(self):
        self.population = [Particle(position=np.random.randn(2))
                           for _ in range(self.population_size)]

        self.best_fitness = []
        self.mean_fitness = []

    def execute(self):
        for interaction in range(self.interations_size):
            fitness = []
            for particle in self.population:
                particle.fitness = self.cost_function(particle.position[0], particle.position[1])
                fitness.append(particle.fitness)
            #  = [particle.fitness for particle in self.population]
            rank = sorted(self.population, key=lambda x: x.fitness, reverse=False)
            self.population = rank

            # print(f"interaction: {interaction + 1} best fitness: {round(rank[0].fitness, 4)}")

            for i, particle in zip(range(1, int(self.selection_rate * self.population_size)), self.population):
                # particle.fitness = self.cost_function(particle.position[0], particle.position[1])
                # fitness.append(particle.fitness)
                # cloning_number = round((self.cloning_rate*self.population_size)/rank[i].fitness)
                cloning_number = round((self.cloning_rate * self.population_size) / i)

                # fitness_ind = (1 - ((rank[i].fitness - 1)/(self.population_size - 1)))
                fitness_ind = (1 - ((i - 1) / (self.population_size - 1)))
                for j in range(cloning_number):
                    clone = deepcopy(rank[j])
                    alpha = self.mutation_range * np.e ** (-fitness_ind)
                    self.hypermutate(clone, alpha)
                    clone_fitness = self.cost_function(clone.position[0], clone.position[1])
                    if clone_fitness < rank[j].fitness:
                        rank[j] = clone

            for i in range((int(self.selection_rate * self.population_size)), self.population_size):
                self.population[i] = Particle(position=np.random.randn(2))

            self.best_fitness.append(round(np.array(fitness).min(), 3))
            self.mean_fitness.append(round(np.array(fitness).mean(), 3))

        # t = [i for i in range(30)]
        # plt.figure(figsize=(10, 5))
        # plt.grid()
        # plt.plot(t, self.best_fitness, 'b')
        # plt.title(f"Best Features P_{440} F_{2} C_{0.9} Without Oposition")
        # plt.savefig(f"images2/clonalg Best Features P_{440} S_{0.9} M_{0.9} C_{0.5}_levy_cost.png")
        #
        # plt.figure(figsize=(10, 5))
        # plt.grid()
        # plt.plot(t, self.mean_fitness, 'b')
        # plt.title(f"Mean Features P_{440} S_{0.9} M_{0.9} C_{0.5}")
        # plt.savefig(f"images2/clonalg Mean Features P_{440} S_{0.9} M_{0.9} C_{0.5}_levy_cost.png")
        return (round(np.array(self.best_fitness).min(), 4),
                round(np.array(self.mean_fitness).min(), 3))

    def hypermutate(self, clone, alpha):
        for i in range(2):
            if np.random.random() < alpha:
                clone.position[i] = np.random.randn()


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


    boxplot_data = np.empty((30, 0))
    boxplot_data_mean = np.empty((30, 0))

    for selec in [440]:
        best_f = []
        mean_f = []
        for _ in range(1):
            clonalg = ClonalG(population_size=selec,
                              cloning_rate=0.5,
                              mutation_range=0.9,
                              interations_size=30,
                              selection_rate=0.9,
                              cost_function=levy_cost
                              )
            best_fitness, mean_fitness = clonalg.execute()
            best_f.append(best_fitness)
            mean_f.append(mean_fitness)
        boxplot_data = np.hstack((boxplot_data, np.array(best_f).reshape(-1, 1)))
        boxplot_data_mean = np.hstack((boxplot_data_mean, np.array(mean_f).reshape(-1, 1)))

    plt.figure(figsize=(10, 5))
    plt.title(f"Mean Fitness")
    df2 = pd.DataFrame(boxplot_data_mean, columns=['P(220)', 'P(440)', 'P(880)'])
    boxplot2 = df2.boxplot()
    fig2 = boxplot2.get_figure()
    fig2.savefig('boxplot_mean_levy_cost_etapa4.png')

    plt.figure(figsize=(10, 5))
    plt.title(f"Best Fitness")
    df = pd.DataFrame(boxplot_data, columns=['P(220)', 'P(440)', 'P(880)'])
    boxplot = df.boxplot()
    fig = boxplot.get_figure()
    fig.savefig('boxplot_best_levy_cost_etapa4.png')
