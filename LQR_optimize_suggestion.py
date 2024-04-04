import numpy as np
import scipy.linalg as la

# Definição do sistema de espaço de estados
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([[1, 0], [0, 1]])
D = np.array([[0], [0]])

# Dimensões do sistema
n_states = A.shape[0]
n_inputs = B.shape[1]


# Função para calcular a função custo do controlador LQR
def lqr_cost(Q, R):
    # Resolver a equação de Riccati para obter a matriz de ganho L ótima
    P = la.solve_continuous_are(A, B, Q, R)

    # Calcular a função custo
    return np.trace(P)


# Função de aptidão para o algoritmo genético
def fitness_function(chromosome):
    # Decodificar cromossomo para matrizes de ponderação Q e R
    Q = chromosome[:n_states ** 2].reshape((n_states, n_states))
    R = chromosome[n_states ** 2:].reshape((n_inputs, n_inputs))

    # Garantir que Q seja simétrica
    Q = (Q + Q.T) / 2

    # Calcular a função custo usando LQR
    return -lqr_cost(Q, R)  # Queremos minimizar a função custo, por isso multiplicamos por -1


# Função de mutação para o algoritmo genético
def mutate(chromosome, mutation_rate=0.1):
    mutated_chromosome = chromosome.copy()
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            mutated_chromosome[i] += np.random.normal(0, 1)  # Mutação gaussiana
    return mutated_chromosome


# Algoritmo genético simples
def genetic_algorithm(population_size, chromosome_length, generations):
    # Inicialização da população
    population = np.random.randn(population_size, chromosome_length)

    # Loop das gerações
    for gen in range(generations):
        # Avaliação da aptidão
        fitness = np.array([fitness_function(chromosome) for chromosome in population])

        # Seleção dos pais (torneio)
        selected_indices = np.random.choice(population_size, size=population_size, replace=True)
        parents = population[selected_indices]

        # Crossover (um ponto de corte)
        crossover_point = np.random.randint(0, chromosome_length, size=population_size)
        offspring = np.array([np.concatenate((parents[i][:crossover_point[i]], parents[(i + 1) % population_size][crossover_point[i]:]))
                              for i in range(population_size)])

        # Mutação
        offspring = np.array([mutate(chromosome) for chromosome in offspring])

        # Substituição da população antiga pela nova
        population = offspring

        # Mostrar a melhor aptidão a cada geração
        print("Geração:", gen, "Melhor Aptidão:", -np.max(fitness))  # Invertemos o sinal para maximizar

    # Encontrar o melhor cromossomo
    best_index = np.argmax(fitness)
    best_chromosome = population[best_index]

    # Decodificar o melhor cromossomo
    Q = best_chromosome[:n_states ** 2].reshape((n_states, n_states))
    R = best_chromosome[n_states ** 2:].reshape((n_inputs, n_inputs))

    return Q, R


# Parâmetros do algoritmo genético
population_size = 50
chromosome_length = n_states ** 2 + n_inputs ** 2  # Tamanho do cromossomo
generations = 100

# Execução do algoritmo genético
best_Q, best_R = genetic_algorithm(population_size, chromosome_length, generations)
print("Matriz de Ponderação Q ótima:\n", best_Q)
print("Matriz de Ponderação R ótima:\n", best_R)
