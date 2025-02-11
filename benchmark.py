import re
import csv
from collections import defaultdict

def parse_benchmark(log_text):
    benchmark_pattern = re.compile(r'BENCHMARK (\d+) \"([^"]+)\" {')
    timer_pattern = re.compile(r'Timer ([^;]+);\s+VK\s+(\d+); CPU\s+(\d+);')
    
    benchmarks = []
    
    benchmark_sections = re.split(benchmark_pattern, log_text)[1:]
    
    for i in range(0, len(benchmark_sections), 3):
        benchmark_id = int(benchmark_sections[i])
        benchmark_name = benchmark_sections[i+1].strip()
        benchmark_data = benchmark_sections[i+2]
        
        timers = {}
        for match in timer_pattern.finditer(benchmark_data):
            stage = match.group(1).strip()
            vk_time = int(match.group(2))
            cpu_time = int(match.group(3))
            timers[stage] = {'VK': vk_time, 'CPU': cpu_time}
        
        benchmarks.append((benchmark_id, benchmark_name, timers))
    
    return benchmarks

def save_to_csv(benchmarks, filename="benchmark_results.csv"):
    stages = {stage for _, _, timers in benchmarks for stage in timers}
    fieldnames = ["Benchmark ID", "Benchmark Name"] + [f"{stage} VK" for stage in stages] + [f"{stage} CPU" for stage in stages]
    
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for benchmark_id, benchmark_name, timers in benchmarks:
            row = {"Benchmark ID": benchmark_id, "Benchmark Name": benchmark_name}
            for stage in stages:
                row[f"{stage} VK"] = timers.get(stage, {}).get("VK", "N/A")
                row[f"{stage} CPU"] = timers.get(stage, {}).get("CPU", "N/A")
            writer.writerow(row)

# Example usage
if __name__ == "__main__":
    with open("_benchmark/benchmark.log", "r", encoding="utf-8") as file:
        log_text = file.read()
    
    results = parse_benchmark(log_text)
    save_to_csv(results, "_benchmark/benchmark.csv" )
    print("CSV file saved as benchmark_results.csv")
