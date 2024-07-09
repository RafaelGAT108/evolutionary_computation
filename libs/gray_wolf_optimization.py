from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
import time
import multiprocessing
@dataclass
class Wolf:
    position: np.ndarray
    fitness: float | None = None


@dataclass
class GrayWolfOptimization:
    cost_function: callable
    population: list[Wolf] = field(init=False)
    population_size: int
    interations_size: int

    def __post_init__(self):
        self.population = [Wolf(position=np.random.randn(2))
                           for _ in range(self.population_size)]

        self.alpha = self.beta = self.delta = None

    def calculate_x(self, a, top_wolf, wolf):
        r1, r2 = np.random.rand(2)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2

        Delta = abs(C1 * top_wolf.position - wolf.position)
        X = self.alpha.position - A1 * Delta
        return X

    def execute(self):
        for gen in range(self.interations_size):

            for wolf in self.population:
                wolf.fitness = self.cost_function(wolf.position[0], wolf.position[1])

            self.population = sorted(self.population, key=lambda x: x.fitness)
            self.alpha, self.beta, self.delta = self.population[:3]

            # print(
            #     f"int: {gen + 1}, posição do alpha: {[round(float(v), 3) for v in self.alpha.position]}, "
            #     f"posição: {round(self.alpha.fitness, 3)}")

            a = 2 - gen * (2 / self.interations_size)

            for i, wolf in enumerate(self.population):
                if i <= 1:
                    continue
                X1 = self.calculate_x(a=a, top_wolf=self.alpha, wolf=wolf)
                X2 = self.calculate_x(a=a, top_wolf=self.beta, wolf=wolf)
                X3 = self.calculate_x(a=a, top_wolf=self.delta, wolf=wolf)

                wolf.position = (X1 + X2 + X3) / 3.0

        return self.alpha


def process_step(equation):
    initial = time.time()
    best_individuals = []
    for _ in range(30):

        gray = GrayWolfOptimization(cost_function=equation,
                                    population_size=100,
                                    interations_size=1000)

        alpha = gray.execute()
        best_individuals.append(alpha)

    best_individuals = sorted(best_individuals, key= lambda x: x.fitness)

    final = time.time()
    print("-" * 100)
    print(f"TEMPO DE 1000 ITERAÇÕES DO {equation}: {round(final - initial, 2)} seg")
    # print(f"posição do alpha: {[round(float(v), 3) for v in alpha.position]}, "
    #       f"fitness: {round(alpha.fitness, 3)}")

    print(f"Melhor: {round(best_individuals[0].fitness, 3)}, "
          f"Pior: {round(best_individuals[-1].fitness, 3)}, "
          f"Média: {round(np.mean([f.fitness for f in best_individuals]), 3)}, "
          f"Mediana: {round(np.median([f.fitness for f in best_individuals]), 3)}, "
          f"Desvio: {round(np.std([f.fitness for f in best_individuals]), 3)}")


if __name__ == "__main__":
    functions = {
        'goldstein_price_cost': goldstein_price_cost,
        'branin_cost': branin_cost,
        'beale_cost': beale_cost,
        'michalewicz_cost': michalewicz_cost,
        'easom_cost': easom_cost,
        'shubert_cost': shubert_cost,
        'schwefel_cost': schwefel_cost,
        'schaffer_4_cost': schaffer_4_cost,
        'schaffer_2_cost': schaffer_2_cost,
        'rastrigin_function': rastrigin_function,
        'levy_13_cost': levy_13_cost,
        'levy_cost': levy_cost,
        'langermann_cost': langermann_cost,
        'holder_table_cost': holder_table_cost,
        'griewank_cost': griewank_cost,
        'eggholder_cost': eggholder_cost,
        'drop_cost': drop_cost,
        'cross_cost': cross_cost,
        'bukin_cost': bukin_cost,
        'ackley_cost': ackley_cost,
        'jong_cost': jong_cost
    }
    list_functions = functions.values()

    # initial_total = time.time()
    # with multiprocessing.Pool(6) as executor:
    #     executor.map(process_step, list_functions)
    #
    # final_total = time.time()
    # print(f"TEMPO TOTAL PARALELO: {round(final_total - initial_total, 2)} seg \n\n")

    initial_total = time.time()
    for equation_name, equation in functions.items():
        initial = time.time()

        best_individuals = []
        for _ in range(30):
            gray = GrayWolfOptimization(cost_function=equation,
                                        population_size=100,
                                        interations_size=1000)

            alpha = gray.execute()
            best_individuals.append(alpha)

        best_individuals = sorted(best_individuals, key=lambda x: x.fitness)

        final = time.time()
        print("-" * 100)
        print(f"30 Execuções de 1000 ITERAÇÕES DO {equation}: {round(final - initial, 2)} seg")
        # print(f"posição do alpha: {[round(float(v), 3) for v in alpha.position]}, "
        #       f"fitness: {round(alpha.fitness, 3)}")

        print(f"Melhor: {round(best_individuals[0].fitness, 3)}, "
              f"Pior: {round(best_individuals[-1].fitness, 3)}, "
              f"Média: {round(np.mean([f.fitness for f in best_individuals]), 3)}, "
              f"Mediana: {round(np.median([f.fitness for f in best_individuals]), 3)}, "
              f"Desvio: {round(np.std([f.fitness for f in best_individuals]), 3)}")
    final_total = time.time()

    print(f"TEMPO TOTAL SEQUENCIAL: {round(final_total - initial_total, 2)} seg")
