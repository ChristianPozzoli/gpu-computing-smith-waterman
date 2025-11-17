import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import NullLocator, ScalarFormatter

filename = "results_17112025_190448.csv"
df = pd.read_csv(filename)

gpu_algorithms = ['cpu', 'parallel', 'smem', 'warp', 'warp_smem']
df = df[df['algorithm'].isin(gpu_algorithms)]
#df = df[~df['input_size'].isin([1000, 5000])]

df = df.groupby(['algorithm', 'input_size'])['time_ms'].min().reset_index() 

cpu_baseline = df[df['algorithm'] == 'cpu'][['input_size', 'time_ms']]
cpu_baseline = cpu_baseline.rename(columns={'time_ms': 'cpu_Time'})

df_speedup = pd.merge(df, cpu_baseline, on='input_size')
df_speedup['speedup'] = df_speedup['cpu_Time'] / df_speedup['time_ms']

df_speedup_gpu = df_speedup[df_speedup['algorithm'] != 'cpu'].copy()

plt.figure(figsize=(10, 6))

ax = sns.lineplot(
    data=df_speedup_gpu,
    x='input_size',
    y='speedup',
    hue='algorithm',
    marker='o',
    estimator='min'
)

#ax.set_xscale('log')
#ax.set_yscale('log')

ax.yaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(ScalarFormatter())

# Baseline a y=1 per riferimento
#ax.axhline(1, color='red', linestyle='--', label='Baseline CPU (1x)')

ax.set_xlabel("Input Size (Scala Log)")
ax.set_ylabel("Speedup (Scala Log)")
ax.set_title("Confronto Speedup su Baseline CPU")
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(title="Ottimizzazione")
plt.tight_layout()
plt.show()