import subprocess
import csv

def run_profiler(output_csv_l1_enabled, output_csv_l1_disabled):
    command = './cache_line_profiler' # Command to run the profiler named "cache_line_profiler"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    output_lines = output.decode('utf-8').split('\n')

    # Initialize CSV writers for both configurations
    csvfile_l1_enabled = open(output_csv_l1_enabled, 'w', newline='')
    csvfile_l1_disabled = open(output_csv_l1_disabled, 'w', newline='')
    fieldnames = ['Stride', 'Iteration', 'Time (cycles)']
    writer_l1_enabled = csv.DictWriter(csvfile_l1_enabled, fieldnames=fieldnames)
    writer_l1_disabled = csv.DictWriter(csvfile_l1_disabled, fieldnames=fieldnames)
    writer_l1_enabled.writeheader()
    writer_l1_disabled.writeheader()

    current_config = None
    current_stride = None
    writer = None

    for line in output_lines:
        if 'L1 Enabled' in line:
            current_config = 'L1'
            writer = writer_l1_enabled
        elif 'L1 Disabled' in line:
            current_config = 'L2'
            writer = writer_l1_disabled
        elif 'Completed stride' in line:
            parts = line.split(' ')
            current_stride = parts[2]
        elif line.startswith("Iteration") and current_config and current_stride:
            parts = line.split(':')
            if len(parts) == 3:
                iteration = parts[1].strip()
                time = parts[2].strip().split(' ')[0]  # Split and get the number before 'cycles'
                writer.writerow({'Stride': current_stride, 'Iteration': iteration, 'Time (cycles)': time})

    csvfile_l1_enabled.close()
    csvfile_l1_disabled.close()
    print(f"CSV files created: {output_csv_l1_enabled} and {output_csv_l1_disabled}")

# Run the profiler and record the output
run_profiler('cache_line_profiler_record_l1_enabled.csv', 'cache_line_profiler_record_l1_disabled.csv')
