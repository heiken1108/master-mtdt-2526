import pandas as pd
import numpy as np

def z_normalize(df: pd.DataFrame, col: str) -> pd.DataFrame:
	df_copy = df.copy()
	mean = df_copy[col].mean()
	std = df_copy[col].std()
	if std == 0 or pd.isna(std):
		df_copy[col] = 0
	else:
		df_copy[col] = (df_copy[col] - mean) / std
	return df_copy

def pairwise_similarity_stepwise(dfs, value_col, id_col, symmetric_matrix=True): #symmetric_matrix = True gir halvparten regneoperasjoner, men vanskeligere å finne neste høyeste verdier for en av id-en
	n = len(dfs)
	sim_matrix = np.zeros((n,n))

	ids = [df[id_col].iloc[0] for df in dfs]

	for i in range(n):
		j_range = range(n) if symmetric_matrix else range(i, n)
		for j in j_range:
			seq_i = dfs[i][value_col].values
			seq_j = dfs[j][value_col].values

			max_steps = min(len(seq_i), len(seq_j))
			seq_i = seq_i[:max_steps]
			seq_j = seq_j[:max_steps]

			if max_steps == 0 or np.std(seq_i) == 0 or np.std(seq_j) == 0:
				sim_matrix[i, j] = 0
			else:
				sim_matrix[i, j] = np.corrcoef(seq_i, seq_j)[0, 1]
	
	return pd.DataFrame(sim_matrix, index=ids, columns=ids)