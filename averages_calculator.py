import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 24})

# Load the CSV file
file_path = 'averages_cut.csv'  # Update with the correct path if needed
data = pd.read_csv(file_path)

# Sort the data by the sum of all weighted averages in ascending order
data['Total'] = data[['N', 'L', 'A', 'V']].sum(axis=1)
data_sorted = data.sort_values(by='Total', ascending=True)
samples_sorted = data_sorted['Sample']
values_sorted = data_sorted[['N', 'L', 'A', 'V']].values

# Define new labels for the legend
legend_labels = ['Number', 'Length', 'Area', 'Volume']

# Set up x-axis positions and bar width
x_sorted = np.arange(len(samples_sorted))
width = 0.22  # Width of the bars

# Define custom colors for the metrics
colors = ['black', 'red', 'green', 'blue']

# Plotting the vertical bar chart
fig, ax = plt.subplots(figsize=(16.8, 8))  # Wider plot to accommodate labels

# Creating vertical bars for each metric with custom colors
for i, (metric, color) in enumerate(zip(legend_labels, colors)):
    ax.bar(x_sorted + i * width, values_sorted[:, i], width=width, label=metric, color=color)

# Adding labels and formatting
ax.set_ylabel('Weighted Averages (µm)')
ax.set_xticks(x_sorted + width * (len(legend_labels) - 1) / 2)
ax.set_xticklabels(samples_sorted, rotation=45, ha='right')
ax.legend(title='Weighting', loc='upper left')

# # Apply log scaling to the y-axis
# ax.set_yscale('log')

# Optional: Set specific y-ticks if you want to customize them
# ax.set_yticks([1, 10, 100, 1000])
# ax.set_yticklabels(['1', '10', '100', '1000'])

# Show the plot
plt.grid(axis='y')
plt.tight_layout()
output_path = os.path.join('figures/', "averages_vertical.png")
# plt.savefig(output_path, dpi=300)
plt.show()
