import scipy.io
import pandas as pd
import numpy as np
mat_file = '../wet_asphalt_90kph.mat'
mat_data = scipy.io.loadmat(mat_file)


for key, value in mat_data.items():
    if isinstance(value, (np.ndarray, list)):
        np.savetxt(f"{key}.txt", value, delimiter="\t", fmt="%s")

for key, value in mat_data.items():
    if isinstance(value, (np.ndarray, list)):
        df = pd.DataFrame(value)
        df.to_csv(f"{key}.csv", index=False)

print("Conversão concluída com sucesso!")