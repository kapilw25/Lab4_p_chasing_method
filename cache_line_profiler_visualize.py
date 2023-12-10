import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('cache_line_profiler_record.csv')

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(df['Iteration'], df['Time (cycles)'], marker='o', linestyle='-', color='b')

# Adding labels and title
plt.xlabel('Iteration')
plt.ylabel('Time (cycles)')
plt.title('Per-Access Timing Data Analysis')

# Save the figure
plt.savefig('cache_line_profiler_timing_analysis.png', dpi=300)
# plt.show()  # If you want to display the plot as well

print("Timing analysis PNG file created: cache_line_profiler_timing_analysis.png")
