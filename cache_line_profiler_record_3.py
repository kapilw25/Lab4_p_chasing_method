import subprocess
import csv

def run_profiler(output_csv):
    command = './cache_line_profiler'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    output_lines = output.decode('utf-8').split('\n')

    csvfile = open(output_csv, 'w', newline='')
    fieldnames = ['CacheConfig', 'Stride', 'Iteration', 'Time (cycles)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    current_config = None

    for line in output_lines:
        parts = line.split(',')
        if 'CacheConfig' in parts[0]:
            current_config = parts[1]
        elif 'Stride' in parts[0] and current_config:
            stride = parts[1]
            iteration = parts[3]
            time = parts[5]
            writer.writerow({'CacheConfig': current_config, 'Stride': stride, 'Iteration': iteration, 'Time (cycles)': time})

    csvfile.close()
    print(f"CSV file created: {output_csv}")

run_profiler('cache_line_profiling_data.csv')
