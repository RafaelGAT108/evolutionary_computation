from mggp import mggpElement, mggpEvolver
import numpy as np
import random
import pandas as pd
import time
#from numba import jit, njit
import cProfile
import pstats

def training_data():
    path_data = "datahydraulic/dynamic_ident.txt"
    data = pd.read_csv(path_data, sep=" ", header=None)
    u = data[0].values
    y = data[1].values
    return y, u


def evaluate(ind):
    try:
        element.compile_model(ind)
        if random.random() < OLSPB:
            element.ols(ind, 1e-7, y, u)

        theta = element.ls_extended(ind, y, u)
        return element.score_osa(ind, theta, y, u),

        # return element.score_freeRun(ind, theta, y, u),
    except np.linalg.LinAlgError as e:
        return np.inf,


element = mggpElement()
# element.setPset(maxDelay=5)
element.renameArguments({"ARG0": "y1", "ARG1": "u1"})
mggp = mggpEvolver(popSize=200, CXPB=0.8, MTPB=0.2,
                   n_gen=10, maxHeight=5, maxTerms=30,
                   elite=10, element=element)

y, u = training_data()
OLSPB = 0.2
k = 500
seed = None
with cProfile.Profile() as profile:

    for i in range(5):
        inicio = time.time()
        hof, log = mggp.run(evaluate=evaluate, seed=seed)
        best = element.model2List(hof[0])
        element.save("fileName.pkl", best)
        seed = hof.items
        fim = time.time()
        print(f"tempo de execução da {i+1} geração: {round(fim - inicio, 2)} segundos")

results = pstats.Stats(profile)
results.sort_stats(pstats.SortKey.TIME)
results.print_stats()


