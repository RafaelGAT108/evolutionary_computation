from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Particle:
    velocity: np.ndarray
    position: np.ndarray

    max_x: float = 10
    min_x: float = -10
    fitness: float | None = None
    best_global_position: np.ndarray | None = None
    best_personal_position: np.ndarray | None = None

    def __post_init__(self):
        self.best_personal_position = self.position
        self.best_global_position = self.position


@dataclass
class ParticleSwarmOptimization:
    generations_size: int
    with_update_w: bool
    size_particles: int
    cost_function: callable
    with_constriction_factor: bool
    particles: list[Particle] = field(init=False)

    best_global_position: np.ndarray | None = None
    c1: int = 2
    c2: int = 2
    w: float = 0.7
    w_max: float = 0.9
    w_min: float = 0.4

    def __post_init__(self):
        self.particles = [Particle(velocity=np.random.randn(2),
                                   position=np.random.randn(2))
                          for _ in range(self.size_particles)]

        self.best_fitness = []
        self.mean_fitness = []

    def update_w(self, interation: int, interation_max):
        self.w = self.w_max - interation * ((self.w_max - self.w_min) / interation_max)

    def constriction_factor(self, particle: Particle):
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)

        v_u1 = r1 * self.c1 * (particle.best_personal_position - particle.position)
        v_u2 = r2 * self.c2 * (particle.best_global_position - particle.position)

        phi = 4.1
        k = 1
        x_factor = (2 * k) / (2 - phi - np.sqrt(phi ** 2 - 4 * phi))

        particle.velocity = x_factor * (particle.velocity + v_u1 + v_u2)

    def update_position(self, particle: Particle):
        particle.position = particle.position + particle.velocity

    def update_velocity(self, particle: Particle):
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)

        v_u1 = r1 * self.c1 * particle.best_personal_position - particle.position
        v_u2 = r2 * self.c2 * particle.best_global_position - particle.position
        particle.velocity = self.w * particle.velocity + v_u1 + v_u2

    def execute(self):
        for gen in range(self.generations_size):
            fitness = []
            for particle in self.particles:
                particle.fitness = self.cost_function(particle.position[0], particle.position[1])
                fitness.append(particle.fitness)
                best_personal_fitness = self.cost_function(particle.best_personal_position[0],
                                                           particle.best_personal_position[1])

                if self.best_global_position is None:
                    self.best_global_position = particle.position

                best_global_fitness = self.cost_function(self.best_global_position[0],
                                                         self.best_global_position[1])

                if particle.fitness < best_personal_fitness:
                    particle.best_personal_position = particle.position

                    if particle.fitness < best_global_fitness:
                        self.best_global_position = particle.position

            print(f"Gen: {gen+1} C1: {c1}  C2: {c2}  W: {w}  "
                  f"Best Global Position: {[round(value, 3) for value in self.best_global_position]} "
                  f"Best Global Fitness: {round(best_global_fitness, 4)}")

            for particle in self.particles:
                # Update particles
                if self.with_constriction_factor:
                    self.constriction_factor(particle)

                else:
                    self.update_velocity(particle)

                self.update_position(particle)
            if self.with_update_w:
                self.update_w(interation=gen+1, interation_max=self.generations_size)

            self.best_fitness.append(round(np.min(np.array(fitness)), 3))
            self.mean_fitness.append(round(np.mean(np.array(fitness)), 3))
        print("-"*100)
        return ([round(value, 3) for value in self.best_global_position],
                round(np.min(np.array(self.best_fitness)), 4),
                round(np.min(np.array(self.mean_fitness)), 3))


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


    # c1_values = [1, 2, 3]
    # c2_values = [1, 2, 3]
    # w_values = [.6, .7, .8]

    c1_values = [1, 2, 3]
    c2_values = [3]
    w_values = [.8]

    executions = []
    boxplot_data = np.empty((30, 0))
    boxplot_data_mean = np.empty((30, 0))

    for c1 in c1_values:
        for c2 in c2_values:
            for w in w_values:

                best_f = []
                mean_f = []
                for _ in range(30):

                    pso = ParticleSwarmOptimization(cost_function=shubert_cost,
                                                    generations_size=30,
                                                    size_particles=440,
                                                    c1=c1, c2=c2, w=w,
                                                    with_update_w=True,
                                                    with_constriction_factor=False)

                    best_global_position, best_fitness, mean_fitness = pso.execute()
                    best_f.append(best_fitness)
                    mean_f.append(mean_fitness)

                    executions.append({
                        "C1": c1,
                        "C2": c2,
                        "W": w,
                        "best_global_position": best_global_position,
                        "best_fitness": best_fitness,
                        "mean_fitness": mean_fitness
                        })

                boxplot_data = np.hstack((boxplot_data, np.array(best_f).reshape(-1, 1)))
                boxplot_data_mean = np.hstack((boxplot_data_mean, np.array(mean_f).reshape(-1, 1)))

    plt.figure(figsize=(10, 5))

    plt.title(f"Mean Fitness")
    # TODO: Ajustar os nomes das colunas adequadamente
    df2 = pd.DataFrame(boxplot_data_mean, columns=['P220', 'P440', "P880"])
    boxplot2 = df2.boxplot()
    fig2 = boxplot2.get_figure()
    # fig.title("Mean Values Features")
    fig2.savefig('boxplot_mean_shubert_cost.png')

    plt.figure(figsize=(10, 5))
    # boxplot_data = np.array(boxplot_data)

    plt.title(f"Best Fitness")
    df = pd.DataFrame(boxplot_data, columns=['P220', 'P440', "P880"])
    boxplot = df.boxplot()
    fig = boxplot.get_figure()
    # fig.title("Best Values Features")
    fig.savefig('boxplot_best_shubert_cost.png')

    executions = sorted(executions, key=lambda x: x["best_fitness"])
    executions = pd.DataFrame(executions)
    executions.to_excel("pso_shubert_cost.xlsx", index=False)
