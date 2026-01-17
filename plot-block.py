import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = "results_25112025_125912.csv"
df = pd.read_csv(filename)

gpu_opts = ['parallel', 'smem', 'warp', 'warp_smem']
df = df[df['algorithm'].isin(gpu_opts)]
df = df[~df['input_size'].isin([1000, 5000])]

df = df.sort_values(by=['input_size', 'block_size'])

block_size_ticks = sorted(df['block_size'].unique())


g = sns.relplot(
    data=df,
    x='block_size',
    y='time_ms',
    hue='algorithm',
    col='input_size',
    kind='line',
    marker='o',
    col_wrap=2,
    estimator='min',
    facet_kws={'sharey': False, 'sharex': False}
)

for ax in g.axes.flat:
    ax.set_xscale('log', base=2)
    ax.set_xticks(block_size_ticks)
    ax.set_xticklabels(block_size_ticks)
    ax.xaxis.set_minor_locator(plt.NullLocator())

g.set_axis_labels("Dimensione Blocco (threads, scala log)", "Tempo minimo [ms]")
g.set_titles("Input Size: {col_name}")

sns.move_legend(
    g,
    "upper center",
    bbox_to_anchor=(.5, 1),
    ncol=4,
    title="Ottimizzazione",
    frameon=False,
)

g.figure.subplots_adjust(top=0.92, left=0.08, right=0.98, hspace=0.175, wspace=0.175)

plt.show()