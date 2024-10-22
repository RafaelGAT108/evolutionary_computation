from dataclasses import dataclass, field
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing
from tqdm import tqdm
import concurrent.futures

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
    dim: int = 9
    alpha = beta = delta = None

    def __post_init__(self):
        self.population = [Wolf(position=np.random.randint(2, size=self.dim))
                               for _ in range(self.population_size)]

    def calculate_x(self, a, top_wolf, wolf):
        # r1 = np.random.rand(2)
        r1 = np.random.random(size=self.dim)
        A1 = 2 * a * r1 - a
        C1 = 2 * r1

        Delta = abs(C1 * top_wolf.position - wolf.position)
        X = self.alpha.position - A1 * Delta
        return X

    def execute_original(self):
        for gen in range(self.interations_size):

            for wolf in tqdm(self.population, desc='Calculating Population Fitness'):
                wolf.fitness = self.cost_function(wolf.position)

            self.population = sorted(self.population, key=lambda x: x.fitness)
            # reverse = True Máximiza.
            # self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            self.alpha, self.beta, self.delta = self.population[:3]

            a = 2 - gen * (2 / self.interations_size)

            for i, wolf in enumerate(self.population):
                if i < 1:
                    continue
                X1 = self.calculate_x(a=a, top_wolf=self.alpha, wolf=wolf)
                X2 = self.calculate_x(a=a, top_wolf=self.beta, wolf=wolf)
                X3 = self.calculate_x(a=a, top_wolf=self.delta, wolf=wolf)

                new_position = (X1 + X2 + X3) / 3.0
                sigmoide = 1 / (1 + np.exp(-new_position))

                wolf.position = np.where(sigmoide >= 0.5, 1, 0)
            
            print(f'Fitness: {1 + 0.001*list(self.alpha.position).count(1) - self.alpha.fitness}| Best: {self.alpha.position}')
            # print(f'Fitness: {self.alpha.fitness}| Best: {self.alpha.position}')

        
    def execute(self):
        for gen in range(self.interations_size):
            # Usar ThreadPoolExecutor para paralelizar o cálculo do fitness
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Calcular fitness de todos os lobos em paralelo
                futures = [executor.submit(self.cost_function, wolf.position) for wolf in self.population]
                for wolf, future in tqdm(zip(self.population, futures), desc='Calculating Population Fitness', total=len(self.population)):
                    wolf.fitness = future.result()

            # Ordenar população de acordo com o fitness
            self.population = sorted(self.population, key=lambda x: x.fitness)
            self.alpha, self.beta, self.delta = self.population[:3]

            a = 2 - gen * (2 / self.interations_size)

            for i, wolf in enumerate(self.population):
                if i < 1:
                    continue
                X1 = self.calculate_x(a=a, top_wolf=self.alpha, wolf=wolf)
                X2 = self.calculate_x(a=a, top_wolf=self.beta, wolf=wolf)
                X3 = self.calculate_x(a=a, top_wolf=self.delta, wolf=wolf)

                new_position = (X1 + X2 + X3) / 3.0
                sigmoide = 1 / (1 + np.exp(-new_position))

                wolf.position = np.where(sigmoide >= 0.5, 1, 0)
            
            # print(f'Fitness: {1 + list(self.alpha.position).count(1) - self.alpha.fitness}| Best: {self.alpha.position}')
            print(f'Fitness: {1 + 0.001*list(self.alpha.position).count(1) - self.alpha.fitness}| Best: {self.alpha.position} | features: {list(self.alpha.position).count(1)}')
            
        return self.alpha

if __name__ == '__main__': 

    def evaluate_function(chromossome):
        familias_destino_cucuta = ['ANG', 'BAR COR', 'BAR LIS', 'PLATINA']
        # familias_destino_cucuta = ['ANG', 'BAR COR', 'BAR LIS', 'PLATINA', 'CDO']

        familias_cucuta = [fam for fam, flag in zip(familias_destino_cucuta, chromossome) if flag == 1]
        # familias_bucaramanga = [fam for fam, flag in zip(familias_destino_cucuta, chromossome) if flag == 0]

        tuta_cucuta_leatime_medio = 0.39

        cucuta_vendas = data_cyrgo[data_cyrgo['Origen de la carga'] == 'CY Cúcuta']
        cucuta_vendas = cucuta_vendas[cucuta_vendas['Denominación de posición'].str.startswith(tuple(familias_cucuta))]

        tuta_cucuta_zones = calculate_zones(cucuta_vendas, lead_time_medio=tuta_cucuta_leatime_medio)

        tuta_bucaramanga_leatime_medio = 0.29

        bucaramanga_vendas = data_cyrgo[data_cyrgo['Origen de la carga'].apply(lambda x: x in ['CY Bucaramanga', 'CY Bucaramanga Centr', 'CY Cúcuta'])]
        # bucaramanga_vendas = bucaramanga_vendas[~bucaramanga_vendas['Denominación de posición'].str.startswith(tuple(familias_cucuta))]
        bucaramanga_vendas = bucaramanga_vendas[~((bucaramanga_vendas['Denominación de posición'].str.startswith(tuple(familias_cucuta))) & (bucaramanga_vendas['Origen de la carga'] == 'CY Cúcuta'))]

        tuta_bucaramanga_zones = calculate_zones(bucaramanga_vendas, lead_time_medio=tuta_bucaramanga_leatime_medio)

        total_inventory = round(tuta_cucuta_zones['inventory'].sum() + tuta_bucaramanga_zones['inventory'].sum(), 3)

        return total_inventory

    def evaluate_function(chromossome):
        # Chromossome example: [0, 1, 1, 0, 0, ..., 1, 1, 0] (len == 138)
        # Os valores que são 1 serão as features escolhidas
        # Treinar uma RF recebendo apenas essas features
        # Retornar o inverso da acurácia, ou seja, 1 - Acurácia
        # Minimizar: 1 - acurária
        data = pd.read_csv('FinalDataset.csv')
        Y = data['Label']
        data = data.drop(columns=['Label', 'Start Timestamp', 'End Timestamp'])

        data = data.loc[:, chromossome]

        return total_inventory


    gray = GrayWolfOptimization(cost_function=evaluate_function,
                                population_size=50,
                                dim=138,
                                interations_size=50)

    alpha = gray.execute()