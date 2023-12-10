import pandas as pd

def deduce_cache_line_sizes(input_csv, output_csv):
    # Load the average times data
    df = pd.read_csv(input_csv)

    # Function to find significant jumps in average time, indicating cache line size
    def find_cache_line_size(df, column):
        previous_time = df.iloc[0][column]
        for index, row in df.iterrows():
            current_time = row[column]
            # Check for a significant increase in time (threshold can be adjusted)
            if current_time > previous_time * 1.03:
                return row['Stride'] - 1  # Cache line size is the previous stride
            previous_time = current_time
        return None

    # Deduce cache line sizes
    cache_line_size_l1 = find_cache_line_size(df, 'Time (cycles)_L1_Enabled')
    cache_line_size_l2 = find_cache_line_size(df, 'Time (cycles)_L1_Disabled')

    # Record the results in a new CSV file
    results_df = pd.DataFrame({
        'Cache': ['L1', 'L2'],
        'Cache Line Size (Stride)': [cache_line_size_l1, cache_line_size_l2]
    })
    results_df.to_csv(output_csv, index=False)

# Specify the input and output file names
input_csv = 'average_times_per_stride.csv'
output_csv = 'deduced_cache_line_sizes.csv'

# Run the function
deduce_cache_line_sizes(input_csv, output_csv)

print(f'Cache line sizes deduced and written to {output_csv}')
