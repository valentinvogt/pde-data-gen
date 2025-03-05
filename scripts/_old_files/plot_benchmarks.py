
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data_from_file(file_path):
    data = {'Grid Size': [], 'Duration CPU': [], 'Duration GPU': []}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        grid_size = None
        cpu_times = []
        gpu_times = []
        for line in lines:
            line = line.strip()
            if line.startswith("Grid size:"):
                if grid_size is not None:
                    data['Grid Size'].extend([grid_size] * len(cpu_times))
                    data['Duration CPU'].extend(cpu_times)
                    data['Duration GPU'].extend(gpu_times)
                grid_size = int(line.split(":")[1].strip())
                cpu_times = []
                gpu_times = []
            elif line.isdigit():
                if len(cpu_times) < 10:
                    cpu_times.append(int(line))
                else:
                    gpu_times.append(int(line))
        if grid_size is not None:
            data['Grid Size'].extend([grid_size] * len(cpu_times))
            data['Duration CPU'].extend(cpu_times)
            data['Duration GPU'].extend(gpu_times)
    return pd.DataFrame(data)

def plot_loglog(data):
    avg_duration = data.groupby('Grid Size').mean()
    std_duration = data.groupby('Grid Size').std()
    
    plt.figure(figsize=(12, 6))
    plt.errorbar(avg_duration.index, avg_duration['Duration CPU'], yerr=std_duration['Duration CPU'], fmt='o-', label='CPU')
    plt.errorbar(avg_duration.index, avg_duration['Duration GPU'], yerr=std_duration['Duration GPU'], fmt='o-', label='GPU')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Grid Size')
    plt.ylabel('Duration (ms)')
    plt.title('Brusselator Algorithm Duration vs Grid Size')
    plt.xticks([8, 16, 32, 64, 128, 256, 512], [8, 16, 32, 64, 128, 256, 512])
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("out/brusselator_benchmark.pdf", bbox_inches='tight')

file_path = 'out/benchmark_results.txt'
data = read_data_from_file(file_path)
plot_loglog(data)
