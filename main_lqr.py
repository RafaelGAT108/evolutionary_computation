from libs.genetic_algoritm_lqr import GeneticAlgoritmLQR
import numpy as np
from control.matlab import ss, step
import matplotlib.pyplot as plt
import scipy.linalg as la
np.seterr(divide='ignore', invalid='ignore')


if __name__ == "__main__":
    # Definição do sistema de espaço de estados
    # A = np.array([[0, 1], [-2, -3]])
    # B = np.array([[0], [1]])
    # C = np.array([[1, 0], [0, 1]])
    # D = np.array([[0], [0]])

    A = np.array([[-0.313, 56.7, 0], [-0.0139, -0.426, 0], [0, 56.7, 0]])
    B = np.array([[0.232], [0.0203], [0]])
    C = np.array([[0, 0, 1]])
    D = np.array([0])

    GA = GeneticAlgoritmLQR(
        population_size=50,
        generations_size=10,
        A=A, B=B, C=C, D=D
    )

    best_Q, best_R = GA.fit_model()

    print("Matriz de Ponderação Q ótima:\n", best_Q)
    print("Matriz de Ponderação R ótima:\n", best_R)

    P = la.solve_continuous_are(A, B, best_Q, best_R)
    K = np.linalg.inv(best_R) @ B.T @ P
    sys = ss(A - B * K, B, C, D)

    t = np.linspace(0, 200, 10000)
    y, t = step(sys, t)  # Response to Unitary Step
    y = np.nan_to_num(y, nan=5000)
    plt.figure()
    plt.grid()
    plt.plot(t, y, 'b')
    plt.savefig("final_system_controller.png")

