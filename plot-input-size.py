import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import NullLocator, ScalarFormatter

filename = "results_17112025_190448.csv"
df = pd.read_csv(filename)

gpu_algorithms = ['parallel', 'smem', 'warp', 'warp_smem']
df = df[df['algorithm'].isin(gpu_algorithms)]
df = df[~df['input_size'].isin([1000, 5000])]

input_size_ticks = sorted(df['input_size'].unique())

plt.figure(figsize=(10, 6))

ax = sns.lineplot(
    data=df,
    x='input_size',
    y='time_ms',
    hue='algorithm',
    marker='o',
    estimator='min'

)

#ax.set_xscale('log')
#ax.set_yscale('log')

ax.set_xticks(input_size_ticks)
ax.set_xticklabels(input_size_ticks)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_formatter(ScalarFormatter())

ax.set_title("Performance Media al variare dell'Input Size")
ax.set_xlabel("Input Size")
ax.set_ylabel("Tempo minimo [ms] (su tutti i block size)")
ax.grid(True, which="both", ls="--", alpha=0.5)

plt.legend(title="Algoritmo")
plt.tight_layout()
plt.show()