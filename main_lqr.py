from libs.genetic_algoritm import GeneticAlgoritmLQR
import numpy as np


if __name__ == "__main__":
    # Definição do sistema de espaço de estados
    A = np.array([[0, 1], [-2, -3]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0], [0, 1]])
    D = np.array([[0], [0]])

    GA = GeneticAlgoritmLQR(
        population_size=50,
        generations_size=100,
        A=A, B=B, C=C, D=D
    )

    best_Q, best_R = GA.fit_model()

    print("Matriz de Ponderação Q ótima:\n", best_Q )
    print("Matriz de Ponderação R ótima:\n", best_R)
