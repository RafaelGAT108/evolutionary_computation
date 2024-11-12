from dataclasses import dataclass, field
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing
from tqdm import tqdm
import concurrent.futures 
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy, copy
import gc


@dataclass
class Student:
    position: np.ndarray
    fitness: Union[float, None] = None

@dataclass
class TLBO:
    cost_function: callable
    population_size: int
    iterations_size: int
    dim: int = 9
    population: list[Student] = field(init=False)

    def __post_init__(self):
        # Inicializa a população com vetores binários de features (80% 0s e 20% 1s)
        self.population = [Student(position=np.random.choice([0, 1], size=self.dim, p=[0.80, 0.20]))
                           for _ in range(self.population_size)]

    def teacher_phase(self, student_position, best_solution, mean_solution, tf, student_fitness):
        # Atualiza a posição baseada na fase do professor, usando seleção binária
        diff_mean = np.random.rand(self.dim) * (best_solution - tf * mean_solution)
        result = student_position + diff_mean
        # selected_solution = np.where(np.random.rand(self.dim) < tf, best_solution, mean_solution)
        # result = np.where(student_position == selected_solution, student_position, selected_solution)
        result = np.where(result >= 0.5, 1, 0)
        result_fitness = self.cost_function(result)

        if result_fitness < student_fitness:
            return result

        return student_position
        

    def learner_phase(self):
        new_population = deepcopy(self.population)

        def update_position(i):
            partner = np.random.randint(0, self.population_size)
            if i != partner:
                
                if self.population[i].fitness < self.population[partner].fitness:

                    diff = self.population[i].position - self.population[partner].position
                    new_position = self.population[i].position + np.random.rand(self.dim) * diff

                else:
                    diff = self.population[partner].position - self.population[i].position 
                    # new_position = self.population[partner].position + np.random.rand(self.dim) * diffnew_position = self.population[partner].position + np.random.rand(self.dim) * diff
                    new_position = self.population[i].position + np.random.rand(self.dim) * diff
                
                result = np.where(new_position >= 0.5, 1, 0) 
                result_fitness = self.cost_function(result) 

                if result_fitness < new_population[i].fitness:
                    new_population[i].position = result

            return i

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(update_position, i) for i in range(self.population_size)]
            for _ in concurrent.futures.as_completed(futures):
                pass

        return new_population

    def execute(self):
        best_overall_student = None
        mean_fitness = []
        mean_features = []

        for iter in range(self.iterations_size):
            print(f'Iteration: {iter + 1}')
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.cost_function, student.position) for student in self.population]
                for student, future in tqdm(zip(self.population, futures), desc='Calculating Population Fitness', total=len(self.population)):
                    student.fitness = future.result()
                executor.shutdown(wait=True)

            self.population = sorted(self.population, key=lambda x: x.fitness)
            best_student = self.population[0]
            if best_overall_student is None or best_student.fitness < best_overall_student.fitness:
                best_overall_student = deepcopy(best_student)

            population_positions = np.array([s.position for s in self.population])
            mean_student_position = np.mean(population_positions, axis=0)
            tf = np.random.randint(1, 2)

            with ProcessPoolExecutor() as exect:
                futures = [exect.submit(self.teacher_phase, student.position, 
                                        best_student.position, 
                                        mean_student_position, 
                                        tf,
                                        student.fitness) for student in self.population]
                for student, future in zip(self.population, futures):
                    student.position = future.result()

            self.population = self.learner_phase()
            self.population[-1] = deepcopy(best_overall_student)
            print(f'Fitness: {round(best_overall_student.fitness, 3)} | features: {list(best_overall_student.position).count(1)}')

            mean_fitness.append(np.mean([student.fitness for student in self.population]))
            mean_features.append(np.mean([list(student.position).count(1) for student in self.population]))

        self.population = sorted(self.population, key=lambda x: x.fitness)
        return best_overall_student, mean_fitness, mean_features
