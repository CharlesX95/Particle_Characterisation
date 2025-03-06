import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


plt.rcParams.update({'font.size': 24})

# Load the CSV file
file_path = 'averages_cut.csv'  # Update with the correct path if needed
data = pd.read_csv(file_path)

# Sort the data by the sum of all weighted averages in descending order
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
colors = ['black','red','green', 'blue']

# Plotting the horizontal bar chart
fig, ax = plt.subplots(figsize=(8, 16))

# Creating horizontal bars for each metric with custom colors
for i, (metric, color) in enumerate(zip(legend_labels, colors)):
    ax.barh(y_sorted_desc + i * width, values_sorted_desc[:, i], height=width, label=metric, color=color)

# Adding labels and formatting
ax.set_xlabel('Weighted Averages (µm)')
ax.set_yticks(y_sorted_desc + width * (len(legend_labels) - 1) / 2)
ax.set_yticklabels(samples_sorted_desc)
ax.legend(title='Weighting', loc='lower right')

# # Apply log scaling to the x-axis
#ax.set_xscale('log')
#ax.set_xticks()  # Set the ticks at desired intervals
#ax.set_xticklabels(['1', '10', '100', '1000'])  # Customize tick labels

# Show the plot
plt.grid(axis='x')
plt.tight_layout()
output_path = os.path.join('figures/', f"averages.png")
plt.savefig(output_path, dpi=300)
#plt.show()
