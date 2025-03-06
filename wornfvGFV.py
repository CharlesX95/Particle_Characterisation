import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})

# Load the CSV file
file_path = 'averages.csv'  # Update with the correct path if needed
data = pd.read_csv(file_path)

# Sort the data by the sum of all weighted averages in ascending order
data['Total'] = data[['N', 'L', 'A', 'V']].sum(axis=1)
data_sorted_desc = data.sort_values(by='Total', ascending=True)
samples_sorted_desc = data_sorted_desc['Sample']
values_sorted_desc = data_sorted_desc[['N', 'L', 'A', 'V']].values

# Define new labels for the legend
legend_labels = ['Number', 'Length', 'Area', 'Volume']

# Set up y-axis positions and bar width
y_sorted_desc = np.arange(len(samples_sorted_desc))
width = 0.22  # Width of the bars

# Define custom colors for the metrics
colors = ['black', 'red', 'green', 'blue']

# Plotting the horizontal bar chart
fig, ax = plt.subplots(figsize=(8, 13))

# Creating horizontal bars for each metric with custom colors
for i, (metric, color) in enumerate(zip(legend_labels, colors)):
    ax.barh(y_sorted_desc + i * width, values_sorted_desc[:, i], height=width, label=metric, color=color)

# Adding labels and formatting
ax.set_xlabel('Weighted Averages (µm)')
ax.set_yticks(y_sorted_desc + width * (len(legend_labels) - 1) / 2)
ax.set_yticklabels(samples_sorted_desc)
ax.legend(title='Weighting', loc='lower right')

# Set x-axis ticks with increments of 50
x_max = np.ceil(values_sorted_desc.max() / 100) * 100  # Determine max tick value rounded up to the nearest 50
ax.set_xticks(np.arange(0, x_max + 1, 100))  # Generate ticks from 0 to max with step of 50

# Add grid lines for clarity
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
