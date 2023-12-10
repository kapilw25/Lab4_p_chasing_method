import pandas as pd

def calculate_average_times(csv_input, csv_output):
    # Load the data from the CSV file
    df = pd.read_csv(csv_input)

    # Separate the data based on cache configuration
    df_l1_enabled = df[df['CacheConfig'] == 'L1 Enabled']
    df_l1_disabled = df[df['CacheConfig'] == 'L1 Disabled']

    # Calculate average time per stride for both configurations
    avg_times_l1_enabled = df_l1_enabled.groupby('Stride')['Time (cycles)'].mean().reset_index()
    avg_times_l1_disabled = df_l1_disabled.groupby('Stride')['Time (cycles)'].mean().reset_index()

    # Merge the two dataframes for comparison
    merged_avg_times = pd.merge(avg_times_l1_enabled, avg_times_l1_disabled, on='Stride', suffixes=('_L1_Enabled', '_L1_Disabled'))

    # Write the merged data to a new CSV file
    merged_avg_times.to_csv(csv_output, index=False)

# Specify the input and output file names
input_csv = 'cache_line_profiling_data.csv'
output_csv = 'average_times_per_stride.csv'

# Run the function
calculate_average_times(input_csv, output_csv)

print(f'Average times per stride written to {output_csv}')
