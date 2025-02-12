import re
import csv
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
import argparse
from collections import defaultdict
import matplotlib.patches as mpatches  # Add this import for legend patches

def run_benchmark(executable, benchmark_file, scene_path, output_log):
    command = [os.path.abspath(executable), "-benchmark", os.path.abspath(benchmark_file), scene_path]
    with open(output_log, "w", encoding="utf-8") as log_file:
        subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, shell=True)

def parse_benchmark(log_text, scene_name):
    benchmark_pattern = re.compile(r'BENCHMARK (\d+) \"([^"]+)\" {')
    timer_pattern = re.compile(r'Timer ([^;]+);\s+VK\s+(\d+); CPU\s+(\d+);')
    benchmark_adv_pattern = re.compile(r'BENCHMARK_ADV (\d+) {')
    memory_pattern = re.compile(r'Memory (\w+); Host used\s+(\d+); Device Used\s+(\d+); Device Allocated\s+(\d+);')

    benchmarks = []
    
    # Extract benchmark sections
    benchmark_sections = re.split(benchmark_pattern, log_text)[1:]
    
    benchmark_data = {}
    for i in range(0, len(benchmark_sections), 3):
        benchmark_id = int(benchmark_sections[i])
        benchmark_name = benchmark_sections[i+1].strip()
        benchmark_content = benchmark_sections[i+2]
        
        timers = {}
        for match in timer_pattern.finditer(benchmark_content):
            stage = match.group(1).strip()
            vk_time = int(match.group(2))
            cpu_time = int(match.group(3))
            timers[stage] = {'VK': vk_time, 'CPU': cpu_time}

        benchmark_data[benchmark_id] = {
            "scene": scene_name,
            "id": benchmark_id,
            "name": benchmark_name,
            "timers": timers,
            "memory": {}
        }

    # Extract BENCHMARK_ADV sections
    benchmark_adv_sections = re.split(benchmark_adv_pattern, log_text)[1:]
    
    for i in range(0, len(benchmark_adv_sections), 2):
        benchmark_id = int(benchmark_adv_sections[i])
        benchmark_content = benchmark_adv_sections[i+1]

        memory_data = {}
        for match in memory_pattern.finditer(benchmark_content):
            memory_type = match.group(1).strip()  # "Scene" or "Rendering"
            host_used = int(match.group(2))
            device_used = int(match.group(3))
            device_allocated = int(match.group(4))

            memory_data[memory_type] = {
                "Host Used": host_used,
                "Device Used": device_used,
                "Device Allocated": device_allocated
            }
        
        # Attach to corresponding BENCHMARK
        if benchmark_id in benchmark_data:
            benchmark_data[benchmark_id]["memory"] = memory_data

    return list(benchmark_data.values())


def save_to_csv(benchmarks, filename="benchmark_results.csv"):
    # Collect all possible timer stages
    stages = sorted({stage for b in benchmarks for stage in b["timers"]})
    
    # Define CSV field names
    fieldnames = ["Scene", "Benchmark ID", "Benchmark Name"]
    fieldnames += [f"{stage} VK" for stage in stages] + [f"{stage} CPU" for stage in stages]
    fieldnames += ["Scene Host Used", "Scene Device Used", "Scene Device Allocated"]
    fieldnames += ["Rendering Host Used", "Rendering Device Used", "Rendering Device Allocated"]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for benchmark in benchmarks:
            row = {
                "Scene": benchmark["scene"],
                "Benchmark ID": benchmark["id"],
                "Benchmark Name": benchmark["name"],
            }
            
            # Add timer values
            for stage in stages:
                row[f"{stage} VK"] = benchmark["timers"].get(stage, {}).get("VK", "N/A")
                row[f"{stage} CPU"] = benchmark["timers"].get(stage, {}).get("CPU", "N/A")

            # Add memory values (default to "N/A" if missing)
            row["Scene Host Used"] = benchmark["memory"].get("Scene", {}).get("Host Used", "N/A")
            row["Scene Device Used"] = benchmark["memory"].get("Scene", {}).get("Device Used", "N/A")
            row["Scene Device Allocated"] = benchmark["memory"].get("Scene", {}).get("Device Allocated", "N/A")

            row["Rendering Host Used"] = benchmark["memory"].get("Rendering", {}).get("Host Used", "N/A")
            row["Rendering Device Used"] = benchmark["memory"].get("Rendering", {}).get("Device Used", "N/A")
            row["Rendering Device Allocated"] = benchmark["memory"].get("Rendering", {}).get("Device Allocated", "N/A")

            writer.writerow(row)

# branding :-)
color_set = {
    "green": "#76B900",
    "dark_green": "#2A7F00", 
    "light_green": "#8BFF00",  
    "black": "#000000",
    "white": "#FFFFFF",
}

def plot_cumulative_histogram_timers(
    benchmarks, 
    title,
    ylabel,
    xlabel,
    pipelines, 
    pipeline_names,  
    stages=["GPU Dist", "GPU Sort", "Rendering"], 
    filename="histogram_timers.png"
):
    scene_groups = defaultdict(list)

    # Group the results by scene and pipeline
    for benchmark in benchmarks:
        scene_name = benchmark["scene"]
        benchmark_name = benchmark["name"]
        timers = benchmark["timers"]
        
        if benchmark_name in pipelines and isinstance(timers, dict):
            scene_groups[scene_name].append((benchmark_name, timers))
    
    all_data = []
    x_labels = []
    width = 0.35  # Base bar width

    # Flatten data for all scenes and pipelines
    for scene, results in scene_groups.items():
        scene_data = {pipeline: {stage: 0 for stage in stages} for pipeline in pipelines}
        for benchmark_name, timers in results:
            for stage in stages:
                scene_data[benchmark_name][stage] += timers.get(stage, {}).get("VK", 0)
        all_data.append(scene_data)
        x_labels.append(scene)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Adjust the space between groups by reducing the range of index values
    index = np.arange(len(x_labels)) * 0.5  # Multiply index by factor to reduce spacing between groups
    num_pipelines = len(pipelines)
    
    # Adjust bar width to ensure space between bars within each group
    bar_width = width * 0.8 / num_pipelines  # Reduce width slightly and divide by number of pipelines

    # Define colors for stages
    stage_colors = [color_set["black"], color_set["dark_green"], color_set["green"]]
    
    # Stack bars
    bottom_values = {pipeline: np.zeros(len(x_labels)) for pipeline in pipelines}

    for i, stage in enumerate(stages):
        for j, pipeline in enumerate(pipelines):
            values = [scene_data[pipeline].get(stage, 0) for scene_data in all_data]

            # Adjust the offset for each bar within a group so that bars are well spaced
            position_offset = (j - (num_pipelines - 1) / 2) * (bar_width + 0.05)  # Add space between bars

            ax.bar(index + position_offset, values, bar_width, color=stage_colors[i], bottom=bottom_values[pipeline])
            bottom_values[pipeline] += np.array(values)

    # Customize plot
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel) 
    ax.set_title(title)   

    # Format x-ticks with pipeline short names
    ax.set_xticks(index)
    ax.set_xticklabels([f"{scene}\n({', '.join(pipeline_names)})" for scene in x_labels], 
                       rotation=45, ha="right")
    
    # Create legend for stages
    legend_handles = [mpatches.Patch(color=stage_colors[i], label=stage) for i, stage in enumerate(stages)]
    ax.legend(handles=legend_handles, title="Stages", loc='upper right')

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Histogram saved as {filename}")

def plot_cumulative_histogram_memory(
    benchmarks, 
    title,
    ylabel,
    xlabel,
    pipelines, 
    pipeline_names,  
    stages=["Scene", "Rendering"], 
    filename="histogram_memory.png"
):
    scene_groups = defaultdict(list)

    # Group the results by scene and pipeline
    for benchmark in benchmarks:
        scene_name = benchmark["scene"]
        benchmark_name = benchmark["name"]
        memory = benchmark["memory"]
        
        if benchmark_name in pipelines and isinstance(memory, dict):
            scene_groups[scene_name].append((benchmark_name, memory))
    
    all_data = []
    x_labels = []
    width = 0.35  # Base bar width

    # Flatten data for all scenes and pipelines
    for scene, results in scene_groups.items():
        scene_data = {pipeline: {stage: 0 for stage in stages} for pipeline in pipelines}
        for benchmark_name, memory in results:
            for stage in stages:
                scene_data[benchmark_name][stage] += memory.get(stage, {}).get("Device Allocated", 0)

        all_data.append(scene_data)
        x_labels.append(scene)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Adjust the space between groups by reducing the range of index values
    index = np.arange(len(x_labels)) * 0.5  # Multiply index by factor to reduce spacing between groups
    num_pipelines = len(pipelines)
    
    # Adjust bar width to ensure space between bars within each group
    bar_width = width * 0.8 / num_pipelines  # Reduce width slightly and divide by number of pipelines

    # Define colors for stages
    stage_colors = [color_set["dark_green"], color_set["green"], color_set["light_green"]]
    
    # Stack bars
    bottom_values = {pipeline: np.zeros(len(x_labels)) for pipeline in pipelines}

    for i, stage in enumerate(stages):
        for j, pipeline in enumerate(pipelines):
            values = [scene_data[pipeline].get(stage, 0) for scene_data in all_data]

            # Adjust the offset for each bar within a group so that bars are well spaced
            position_offset = (j - (num_pipelines - 1) / 2) * (bar_width + 0.05)  # Add space between bars

            ax.bar(index + position_offset, values, bar_width, color=stage_colors[i], bottom=bottom_values[pipeline])
            bottom_values[pipeline] += np.array(values)

    # Customize plot
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel) 
    ax.set_title(title)   

    # Format x-ticks with pipeline short names
    ax.set_xticks(index)
    ax.set_xticklabels([f"{scene}\n({', '.join(pipeline_names)})" for scene in x_labels], 
                       rotation=45, ha="right")
    
    # Create legend for stages
    legend_handles = [mpatches.Patch(color=stage_colors[i], label=stage) for i, stage in enumerate(stages)]
    ax.legend(handles=legend_handles, title="VRAM Cost", loc='upper right')

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Histogram saved as {filename}")


if __name__ == "__main__":    
    executable = os.path.abspath("bin_x64/Release/vk_gaussian_splatting.exe")
    benchmark_file = os.path.abspath("benchmark.txt")
    # Setup argument parsing for the base dataset path
    parser = argparse.ArgumentParser(description="Run benchmarks for 3D scenes.")
    parser.add_argument("dataset_path", type=str, help="Base path to the dataset")
    
    args = parser.parse_args()
    
    # Define the scenes with the relative paths
    scenes = {
        "bicycle 6.13M Splats": "bicycle/bicycle/point_cloud/iteration_30000/point_cloud.ply",
        "bonsai 1.24M Splats": "bonsai/bonsai/point_cloud/iteration_30000/point_cloud.ply",
        "counter 1.22M Splats": "counter/point_cloud/iteration_30000/point_cloud.ply",
        "drjohnson 3.41M Splats": "drjohnson/point_cloud/iteration_30000/point_cloud.ply",
        "flowers 3.64M Splats": "flowers/point_cloud/iteration_30000/point_cloud.ply",
        "garden 5.83M Splats": "garden/point_cloud/iteration_30000/point_cloud.ply",
        "kitchen 1.85M Splats": "kitchen/point_cloud/iteration_30000/point_cloud.ply",
        "playroom 2.55M Splats": "playroom/point_cloud/iteration_30000/point_cloud.ply",
        "room 1.59M Splats": "room/point_cloud/iteration_30000/point_cloud.ply",
        "stump 4.96M Splats": "stump/point_cloud/iteration_30000/point_cloud.ply",
        "train 1.03M Splats": "train/point_cloud/iteration_30000/point_cloud.ply",
        "treehill 3.78M Splats": "treehill/point_cloud/iteration_30000/point_cloud.ply",
        "truck 2.54M Splats": "truck/point_cloud/iteration_30000/point_cloud.ply"
    }
    
    # Build the full paths by combining the dataset path and scene relative paths
    full_scene_paths = {
        scene_name: os.path.join(args.dataset_path, relative_path)
        for scene_name, relative_path in scenes.items()
    }
    
    # Prepare the benchmark directory and output files
    benchmark_dir = "_benchmark"
    os.makedirs(benchmark_dir, exist_ok=True)
    os.chdir(benchmark_dir)

    all_results = []
    for scene_name, scene_path in full_scene_paths.items():
        output_log = f"benchmark_{scene_name.replace(' ', '_')}.log"
        print(f"Running benchmark for {scene_name} at {scene_path}...")
        run_benchmark(executable, benchmark_file, scene_path, output_log)
        
        with open(output_log, "r", encoding="utf-8") as file:
            log_text = file.read()
        
        results = parse_benchmark(log_text, scene_name)
        all_results.extend(results)
    
    save_to_csv(all_results)

    plot_cumulative_histogram_timers(
        all_results, 
        xlabel="Scene",
        ylabel="Cumulative VK Time (microseconds)",
        title="Pipeline Performance Comparison - Mesh vs. Vert",
        pipelines = ["Mesh pipeline", "Vert pipeline"], 
        pipeline_names= ["Mesh", "Vert"],
        stages=["GPU Dist", "GPU Sort", "Rendering"], 
        filename="histogram_shader_timers.png")

    plot_cumulative_histogram_timers(
        all_results, 
        xlabel="Scene",
        ylabel="Cumulative VK Time (microseconds)",
        title="Pipeline Performance Comparison - SH storage formats in float 32, float 16 and uint 8",
        pipelines = ["Mesh pipeline", "Mesh pipeline fp16", "Mesh pipeline uint8"],
        pipeline_names= ["fp32", "fp16", "uint8"],
        stages=["GPU Dist", "GPU Sort", "Rendering"], 
        filename="histogram_format_timers.png")

    plot_cumulative_histogram_memory(
        all_results, 
        xlabel="Scene",
        ylabel="Cumulative VRAM usage (bytes)",
        title="Memory Consumption Comparison - SH storage formats in float 32, float 16 and uint 8",
        pipelines = ["Mesh pipeline", "Mesh pipeline fp16", "Mesh pipeline uint8"],
        pipeline_names= ["fp32", "fp16", "uint8"],
        stages=["Scene", "Rendering"], 
        filename="histogram_format_memory.png")

    print("CSV and histogram generation complete.")
