import pandas as pd
import matplotlib.pyplot as plt

def plot_scatter_from_profiling_data(csv_file, output_file):
    df = pd.read_csv(csv_file)

    # Separate data based on cache configuration
    df_l1_enabled = df[df['CacheConfig'] == 'L1 Enabled']
    df_l1_disabled = df[df['CacheConfig'] == 'L1 Disabled']

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(df_l1_enabled['Stride'], df_l1_enabled['Time (cycles)'], alpha=0.5, label='L1 Enabled')
    plt.scatter(df_l1_disabled['Stride'], df_l1_disabled['Time (cycles)'], alpha=0.5, label='L1 Disabled')
    plt.title('Time (cycles) vs Stride')
    plt.xlabel('Stride')
    plt.ylabel('Time (cycles)')
    plt.legend()
    plt.savefig(output_file)
    plt.close()

def plot_line_from_average_times(csv_file, output_file):
    df_avg = pd.read_csv(csv_file)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df_avg['Stride'], df_avg['Time (cycles)_L1_Enabled'], label='L1 Enabled')
    plt.plot(df_avg['Stride'], df_avg['Time (cycles)_L1_Disabled'], label='L1 Disabled')
    plt.title('Average Time (cycles) per Stride')
    plt.xlabel('Stride')
    plt.ylabel('Average Time (cycles)')
    plt.legend()
    plt.savefig(output_file)
    plt.close()

# File paths
profiling_data_csv = 'cache_line_profiling_data.csv'
average_times_csv = 'average_times_per_stride.csv'

# Generate plots
plot_scatter_from_profiling_data(profiling_data_csv, 'time_vs_stride_scatter.png')
plot_line_from_average_times(average_times_csv, 'average_time_per_stride.png')

print('Plots saved as "time_vs_stride_scatter.png" and "average_time_per_stride.png"')
