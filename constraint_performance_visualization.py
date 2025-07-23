import matplotlib.pyplot as plt
import numpy as np

# Data for each constraint-removed variant
models = ['CCAE_EB', 'CCAE_MB', 'CCAE_ME']
metrics = ['Trendability', 'Robustness', 'Consistency']

# Average values extracted from your dataset
data = {
    'CCAE_EB': [-0.284, 0.902, 0.918],
    'CCAE_MB': [-0.670, 0.922, 0.823],
    'CCAE_ME': [-0.417, 0.956, 0.753]
}

# Set up plot
x = np.arange(len(metrics))  # positions for the metrics
width = 0.2  # smaller bar width

# colors = ['#4B8BBE', '#306998']  # Cool palette

colors = ['#AEC6CF', '#B39EB5', '#4B8BBE']  # Pastel Blue, Orange, Lavender

fig, ax = plt.subplots(figsize=(7.5, 5.0))

# Plot each bar group with tighter spacing and clear labels
rects1 = ax.bar(x - width, data['CCAE_EB'], width, label='Monotonicity', color=colors[0])
rects2 = ax.bar(x, data['CCAE_MB'], width, label='Energy-HI Consistency', color=colors[1])
rects3 = ax.bar(x + width, data['CCAE_ME'], width, label='Bounds', color=colors[2])

# Axis labels and ticks
ax.set_ylabel('Mean Performance')
ax.set_title('Impact of Constraints on CCAE Performance.')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim([-1.0, 1.1])
ax.legend(title='Missing Constraint')

# Annotate bars with performance values
def annotate_bars(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5 if height >= 0 else -15),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

annotate_bars(rects1)
annotate_bars(rects2)
annotate_bars(rects3)

plt.tight_layout()
plt.show()
