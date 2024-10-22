from dataclasses import dataclass, field
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing
from tqdm import tqdm
import concurrent.futures
from copy import deepcopy, copy

@dataclass
class Student:
    position: np.ndarray
    fitness: float | None = None

@dataclass
class TLBO:
    cost_function: callable
    population: list[Student] = field(init=False)
    population_size: int
    iterations_size: int
    dim: int = 9

    def __post_init__(self):
        self.population = [Student(position=np.random.randint(2, size=self.dim))
                               for _ in range(self.population_size)]


    def teacher_phase(self, student_position, best_solution, mean_solution, tf):
        result = student_position + np.random.rand(1, student_position.shape[0]) * (best_solution - tf * mean_solution)
        return result.reshape((self.dim,))


    def learner_phase(self):
        new_population = np.copy(self.population)
        
        def update_position(i):
            partner = np.random.randint(0, self.population_size)
            if i != partner:
                # if self.cost_function(self.population[i].position) < self.cost_function(self.population[partner].position):
                if self.population[i].fitness < self.population[partner].fitness:

                    new_population[i].position += np.random.rand() * (self.population[i].position - self.population[partner].position)
                else:
                    new_population[i].position += np.random.rand() * (self.population[partner].position - self.population[i].position)
            return i

        with concurrent.futures.ThreadPoolExecutor() as executor:

            futures = [executor.submit(update_position, i) for i in range(self.population_size)]
            # for _ in tqdm(concurrent.futures.as_completed(futures), desc='Calculating Learner Phase', total=self.population_size):
            for _ in concurrent.futures.as_completed(futures):
                pass

        return new_population


    def execute(self):
        best_overall_student = None

        for iter in range(self.iterations_size):
            print(f'Iteration: {iter + 1}')
            with concurrent.futures.ThreadPoolExecutor() as executor:

                futures = [executor.submit(self.cost_function, student.position) for student in self.population]
                for student, future in tqdm(zip(self.population, futures), desc='Calculating Population Fitness', total=len(self.population)):
                    student.fitness = future.result()
            
            self.population = sorted(self.population, key=lambda x: x.fitness)
            best_student = self.population[0]

            if best_overall_student is None or best_student.fitness < best_overall_student.fitness:
                best_overall_student = copy(best_student)
            
            population_positions = np.array([s.position for s in self.population])
            mean_student_position = np.mean(population_positions, axis=0)

            tf = np.random.randint(1, 3)

            with concurrent.futures.ThreadPoolExecutor() as exect:

                futures = [exect.submit(self.teacher_phase, student.position, 
                                                               best_student.position, 
                                                               mean_student_position, 
                                                               tf) for student in self.population]

                # for student, future in tqdm(zip(self.population, futures), desc='Calculating Teacher Phase', total=len(self.population)):
                for student, future in zip(self.population, futures):
                    student.position = future.result()


            self.population = self.learner_phase()
            
            def sigmoid(value):
                sigmoide = 1 / (1 + np.exp(-value))
                return np.where(sigmoide >= 0.5, 1, 0)

            for student in self.population:
                student.position = np.array([sigmoid(p) for p in student.position])

            self.population[-1] = copy(best_overall_student)
            print(f'------------------- Fitness: {round(best_overall_student.fitness, 3)} | features: {list(best_overall_student.position).count(1)} -------------------\n')
            # print(f'Fitness: {round(best_overall_student.fitness, 3)}| Best: {best_overall_student.position} | features: {list(best_overall_student.position).count(1)}')
            # print(f'best_student.fitness: {best_student.fitness} | best_overall_student.fitness: {best_overall_student.fitness:}')
                

        self.population = sorted(self.population, key=lambda x: x.fitness)
             
        return best_overall_student
