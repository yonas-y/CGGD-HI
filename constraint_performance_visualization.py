import matplotlib.pyplot as plt
import numpy as np

# Define metrics and models
metrics = ['Trendability', 'Robustness', 'Consistency']
models = ['CCAE_EB', 'CCAE_MB', 'CCAE_ME']

# Reference performance (CCAE with all constraints)
reference = [0.489, 0.927, 0.873]

# Performance of each constraint-removed model
data = {
    'CCAE_EB': [0.284, 0.902, 0.918],
    'CCAE_MB': [0.670, 0.922, 0.823],
    'CCAE_ME': [0.417, 0.956, 0.753]
}

# Calculate deltas (Reference - Model Performance)
deltas = {
    model: [ref - val for ref, val in zip(reference, data[model])]
    for model in models
}

# Plot settings
x = np.arange(len(metrics))
width = 0.2
colors = ['#AEC6CF', '#B39EB5', '#4B8BBE']

fig, ax = plt.subplots(figsize=(8.25, 6.25))

# Plot deltas
for i, model in enumerate(models):
    ax.bar(x + (i - 1) * width, deltas[model], width, label=model, color=colors[i])

# Labels and formatting
ax.set_ylabel('Δ Performance (CCAE - Semi-Constrained Model)')
ax.axis(ymin=-0.21, ymax=0.23)
ax.set_title('Impact of constraints on CCAE performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.legend(title = 'Semi-Constrained Models')

# Annotate deltas
def annotate_deltas(values, positions):
    for i, val in enumerate(values):
        ax.annotate(f'{val:.2f}', xy=(positions[i], val),
                    xytext=(0, 5 if val >= 0 else -15), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

# Apply annotation to each bar group
for i, model in enumerate(models):
    positions = x + (i - 1) * width
    annotate_deltas(deltas[model], positions)

plt.tight_layout()
plt.show()
