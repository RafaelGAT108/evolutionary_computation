from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class Particle:
    position: np.ndarray
    fitness: float | None = None


@dataclass
class DifferencialEvolution:
    population: list[Particle] = field(init=False)
    population_size: int
    crossover_rate: float
    interations_size: int
    with_oposition_learning: bool
    cost_function: callable
    F: float

    def __post_init__(self):
        self.population = [Particle(position=np.random.randn(2))
                           for _ in range(self.population_size)]

        self.best_fitness = []
        self.mean_fitness = []

    def execute(self):

        for gen in range(self.interations_size):
            fitness = []
            for i, particle in enumerate(self.population):
                random_index = np.random.choice(np.delete(np.arange(self.population_size), i), size=3, replace=False)
                particle.fitness = self.cost_function(particle.position[0], particle.position[1])
                fitness.append(particle.fitness)

                a, b, c = [self.population[i] for i in random_index]

                index_variable_selected = np.random.choice(2)

                new_particle = Particle(position=np.random.randn(2))
                for j in range(2):
                    if self.with_oposition_learning:
                        new_particle.position[j] = - particle.position[j]
                    else:
                        if np.random.rand(1) <= self.crossover_rate or j == index_variable_selected:
                            new_particle.position[j] = a.position[j] + self.F * (b.position[j] - c.position[j])

                        else:
                            new_particle.position[j] = particle.position[j]

                new_particle.fitness = self.cost_function(new_particle.position[0], new_particle.position[1])
                if new_particle.fitness < particle.fitness:
                    # particle = new_particle
                    self.population[i] = new_particle
                # print(f"Gen: {i + 1}  Best Fitness: {round(np.array(fitness).min(), 3)}")

            self.best_fitness.append(round(np.array(fitness).min(), 3))
            self.mean_fitness.append(round(np.array(fitness).mean(), 3))

        # t = [i for i in range(30)]
        # plt.figure(figsize=(10, 5))
        # plt.grid()
        # plt.plot(t, self.best_fitness, 'b')
        # plt.title(f"Best Features P_{440} F_{2} C_{0.9} Without Oposition")
        # plt.savefig(f"images2/DE Best Features P_{440} F_{2} C_{0.9} without_oposition_shubert_cost.png")
        #
        # plt.figure(figsize=(10, 5))
        # plt.grid()
        # plt.plot(t, self.mean_fitness, 'b')
        # plt.title(f"Mean Features P_{440} F_{2} C_{0.9} Without Oposition")
        # plt.savefig(f"images2/DE Mean Features P_{440} F_{2} C_{0.9} without_oposition_shubert_cost.png")

        # print(f"Best FINAL Fitness: {round(np.array(self.best_fitness).min(), 3)}")
        return (round(np.array(self.best_fitness).min(), 4),
                round(np.array(self.mean_fitness).min(), 3))


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
    for pop in [220, 440, 880]:
        best_f = []
        mean_f = []
        for _ in range(30):
            DE = DifferencialEvolution(population_size=440,
                                       interations_size=30,
                                       crossover_rate=0.9,
                                       cost_function=levy_cost,
                                       with_oposition_learning=False,
                                       F=2)
            best_fitness, mean_fitness = DE.execute()
            best_f.append(best_fitness)
            mean_f.append(mean_fitness)
        boxplot_data = np.hstack((boxplot_data, np.array(best_f).reshape(-1, 1)))
        boxplot_data_mean = np.hstack((boxplot_data_mean, np.array(mean_f).reshape(-1, 1)))

    plt.figure(figsize=(10, 5))

    plt.title(f"Mean Fitness")
    # df2 = pd.DataFrame(boxplot_data_mean, columns=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'])
    df2 = pd.DataFrame(boxplot_data_mean, columns=["P=220", "P=440", "P=880"])
    boxplot2 = df2.boxplot()
    fig2 = boxplot2.get_figure()
    # fig.title("Mean Values Features")
    # fig2.savefig('boxplot_DE_mean_shubert_cost_etapa3.png')
    fig2.savefig('boxplot_DE_mean_levy_cost_etapa4.png')

    plt.figure(figsize=(10, 5))
    # boxplot_data = np.array(boxplot_data)

    plt.title(f"Best Fitness")
    df = pd.DataFrame(boxplot_data, columns=["P=220", "P=440", "P=880"])
    boxplot = df.boxplot()
    fig = boxplot.get_figure()
    # fig.title("Best Values Features")
    # fig.savefig('boxplot_DE_best_shubert_cost_etapa3.png')
    fig.savefig('boxplot_DE_best_levy_cost_etapa4.png')
