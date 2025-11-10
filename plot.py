import pandas as pd
import matplotlib.pyplot as plt

filename = "results_10112025_215617.csv"
input_size_filter = 10000
block_size_filter = 32

df = pd.read_csv(filename)

df = df[df['block_size'].isin([block_size_filter])]
#df = df[df['input_size'].isin([input_size_filter])]
df = df[~df['algorithm'].isin(['parallel_error', 'smem_error', 'warp_error', 'warp_smem_error', 'cpu'])]

df = df.groupby(['algorithm','block_size','input_size'])['time_ms'].agg('mean').reset_index()
print(df)

plt.figure(figsize=(8, 5))

for opt in df['algorithm'].unique():
    subset = df[df['algorithm'] == opt]
    subset = subset.sort_values('input_size')
    plt.plot(subset['input_size'], subset['time_ms'], marker='o', label=opt)

#plt.title(f"Confronto algoritmi GPU - input_size={input_size_filter}")
#plt.xlabel("Dimensione blocco (threads per blocco)")
plt.ylabel("Tempo di computazione [ms]")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Algoritmo")
plt.tight_layout()

plt.show()
