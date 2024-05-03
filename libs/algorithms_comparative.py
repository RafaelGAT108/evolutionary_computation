from libs.differencial_evolution import DifferencialEvolution
from libs.genetic_algorithm_binary import GeneticAlgorithmBinary
import numpy as np
import pandas as pd
from libs.particle_swarm_optimization import ParticleSwarmOptimization
import matplotlib.pyplot as plt
import time


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

    best_fitness_ga = []
    mean_fitness_ga = []
    initial = time.time()
    for i in range(30):
        GA = GeneticAlgorithmBinary(
            population_size=880,
            generations_size=30,
            cost_function=shubert_cost,
            selection_method="tournament",
            mutation_rate=0.05,
            crossover_rate=0.7,
            elitism=True
        )
        _, _, best_f, mean_f = GA.fit_model()
        best_fitness_ga.append(best_f)
        mean_fitness_ga.append(mean_f)
    final = time.time()
    print("-"*100)
    print(f"TEMPO DE 30 EXECUÇÕES DO GA: {final - initial}")
    print("-" * 100)

    boxplot_data = np.hstack((boxplot_data, np.array(best_fitness_ga).reshape(-1, 1)))
    boxplot_data_mean = np.hstack((boxplot_data_mean, np.array(mean_fitness_ga).reshape(-1, 1)))

    best_fitness_pso = []
    mean_fitness_pso = []

    initial = time.time()
    for i in range(30):
        pso = ParticleSwarmOptimization(cost_function=shubert_cost,
                                        interations_size=30,
                                        size_particles=440,
                                        c1=2, c2=2,
                                        with_update_w=True,
                                        with_constriction_factor=True)

        best_global_position, best_fitness, mean_fitness = pso.execute()
        best_fitness_pso.append(best_fitness)
        mean_fitness_pso.append(mean_fitness)
    final = time.time()
    print("-" * 100)
    print(f"TEMPO DE 30 EXECUÇÕES DO PSO: {final - initial}")
    print("-" * 100)

    boxplot_data = np.hstack((boxplot_data, np.array(best_fitness_pso).reshape(-1, 1)))
    boxplot_data_mean = np.hstack((boxplot_data_mean, np.array(mean_fitness_pso).reshape(-1, 1)))

    best_fitness_de = []
    mean_fitness_de = []

    initial = time.time()
    for i in range(30):
        DE = DifferencialEvolution(population_size=440,
                                   interations_size=30,
                                   crossover_rate=0.9,
                                   cost_function=shubert_cost,
                                   with_oposition_learning=False,
                                   F=2)

        best_fitness, mean_fitness = DE.execute()

        best_fitness_de.append(best_fitness)
        mean_fitness_de.append(mean_fitness)
    final = time.time()
    print("-" * 100)
    print(f"TEMPO DE 30 EXECUÇÕES DO DE: {final - initial}")
    print("-" * 100)

    boxplot_data = np.hstack((boxplot_data, np.array(best_fitness_de).reshape(-1, 1)))
    boxplot_data_mean = np.hstack((boxplot_data_mean, np.array(mean_fitness_de).reshape(-1, 1)))

    plt.figure(figsize=(10, 5))
    plt.title(f"Mean Fitness")
    df2 = pd.DataFrame(boxplot_data_mean, columns=['GA', 'PSO', 'DE'])
    boxplot2 = df2.boxplot()
    fig2 = boxplot2.get_figure()

    fig2.savefig('boxplot_mean_shubert_cost_comparative_ga_pso_de.png')

    plt.figure(figsize=(10, 5))
    plt.title(f"Best Fitness")
    df = pd.DataFrame(boxplot_data, columns=['GA', 'PSO', 'DE'])
    boxplot = df.boxplot()
    fig = boxplot.get_figure()

    fig.savefig('boxplot_best_shubert_cost_comparative_ga_pso_de.png')

