from dataclasses import dataclass, field
from typing import Union
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
    min_range: Union[list, float, int]
    max_range: Union[list, float, int]
    alpha = beta = delta = None

    def __post_init__(self):
        if isinstance(self.max_range, Union[float, int]) and isinstance(self.min_range, Union[float, int]):
            self.population = [Wolf(position=np.random.uniform(self.min_range, self.max_range, 2))
                               for _ in range(self.population_size)]

        elif type(self.max_range) is list and type(self.min_range) is list:
            self.population = [Wolf(position=np.array([float(np.random.uniform(self.min_range[0],
                                                                               self.min_range[1], 1)[0]),
                                                       float(np.random.uniform(self.max_range[0],
                                                                               self.max_range[1], 1)[0])]))
                               for _ in range(self.population_size)]

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

            a = 2 - gen * (2 / self.interations_size)

            for i, wolf in enumerate(self.population):
                if i < 1:
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

    best_individuals = sorted(best_individuals, key=lambda x: x.fitness)

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
        'goldstein_price_cost': dict(func=goldstein_price_cost,
                                     range=[-2, 2]),
        'branin_cost': dict(func=branin_cost,
                            range=[[-5, 10], [0, 15]]),
        'beale_cost': dict(func=beale_cost,
                           range=[-4.5, 4.5]),
        'michalewicz_cost': dict(func=michalewicz_cost,
                                 range=[0, np.pi]),
        'easom_cost': dict(func=easom_cost,
                           range=[-100, 100]),
        'shubert_cost': dict(func=shubert_cost,
                             range=[-10, 10]),
        'schwefel_cost': dict(func=schwefel_cost,
                              range=[-500, 500]),
        'schaffer_4_cost': dict(func=schaffer_4_cost,
                                range=[-100, 100]),
        'schaffer_2_cost': dict(func=schaffer_2_cost,
                                range=[-100, 100]),
        'rastrigin_function': dict(func=rastrigin_function,
                                   range=[-5.12, 5.12]),
        'levy_13_cost': dict(func=levy_13_cost,
                             range=[-10, 10]),
        'levy_cost': dict(func=levy_cost,
                          range=[-10, 10]),
        'langermann_cost': dict(func=langermann_cost,
                                range=[0, 10]),
        'holder_table_cost': dict(func=holder_table_cost,
                                  range=[-10, 10]),
        'griewank_cost': dict(func=griewank_cost,
                              range=[-600, 600]),
        'eggholder_cost': dict(func=eggholder_cost,
                               range=[-512, 512]),
        'drop_cost': dict(func=drop_cost,
                          range=[-5.12, 5.12]),
        'cross_cost': dict(func=cross_cost,
                           range=[-10, 10]),
        'bukin_cost': dict(func=bukin_cost,
                           range=[[-15, -5], [-3, 3]]),
        'ackley_cost': dict(func=ackley_cost,
                            range=[-32.768, 32.768]),
        'jong_cost': dict(func=jong_cost,
                          range=[-65.536, 65.536])
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
            gray = GrayWolfOptimization(cost_function=equation['func'],
                                        population_size=100,
                                        interations_size=1000,
                                        max_range=equation['range'][1],
                                        min_range=equation['range'][0])

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
