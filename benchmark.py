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
        
        benchmarks.append((scene_name, benchmark_id, benchmark_name, timers))
    
    return benchmarks

def save_to_csv(benchmarks, filename="benchmark_results.csv"):
    stages = {stage for _, _, _, timers in benchmarks for stage in timers}
    fieldnames = ["Scene", "Benchmark ID", "Benchmark Name"] + [f"{stage} VK" for stage in stages] + [f"{stage} CPU" for stage in stages]
    
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for scene_name, benchmark_id, benchmark_name, timers in benchmarks:
            row = {"Scene": scene_name, "Benchmark ID": benchmark_id, "Benchmark Name": benchmark_name}
            for stage in stages:
                row[f"{stage} VK"] = timers.get(stage, {}).get("VK", "N/A")
                row[f"{stage} CPU"] = timers.get(stage, {}).get("CPU", "N/A")
            writer.writerow(row)

# branding :-)
nvidia_colors = {
    "green": "#76B900",
    "dark_green": "#2A7F00", #"#3C7021",
    "light_green": "#8BFF00",  
    "black": "#000000",
    "white": "#FFFFFF",
}

def plot_cumulative_histogram(benchmarks, filename="histogram.png"):
    pipelines = ["Mesh pipeline", "Vert pipeline"]
    scene_groups = defaultdict(list)
    
    # Group the results by scene and pipeline
    for scene_name, _, benchmark_name, timers in benchmarks:
        if benchmark_name in pipelines:
            scene_groups[scene_name].append((benchmark_name, timers))
    
    if not scene_groups:
        print("No relevant benchmarks found.")
        return
    
    stages = ["GPU Dist", "GPU Sort", "Rendering"]
    all_data = []
    x_labels = []
    width = 0.35  # Bar width (to create space for two bars per scene)
    
    # Flatten the data for all scenes and pipelines
    for scene, results in scene_groups.items():
        scene_data = {"Mesh pipeline": {stage: 0 for stage in stages},
                      "Vert pipeline": {stage: 0 for stage in stages}}  # Initialize data for both pipelines
        for benchmark_name, timers in results:
            for stage in stages:
                scene_data[benchmark_name][stage] += timers.get(stage, {}).get("VK", 0)  # Sum up VK times for each stage
        all_data.append(scene_data)
        x_labels.append(scene)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    index = np.arange(len(x_labels))  # Position of bars (one for Mesh, one for Vert)
    bar_width = width  # Bar width
    spacing = 0.1  # Space between bars for Mesh and Vert
    
    # Apply NVIDIA colors (swap GPU Dist and Rendering)
    stage_colors = [nvidia_colors["black"], nvidia_colors["dark_green"], nvidia_colors["green"]]
    pipeline_colors = [nvidia_colors["green"], nvidia_colors["dark_green"]]  # Green and dark green for Mesh/Vert

    # Plot stacked bars for each pipeline
    bottom_mesh = np.zeros(len(x_labels))  # Start the stacking for Mesh
    bottom_vert = np.zeros(len(x_labels))  # Start the stacking for Vert

    # Plot bars for each stage (stacked for both Mesh and Vert pipelines)
    for i, stage in enumerate(stages):
        # For each scene, plot Mesh and Vert bars
        mesh_values = [scene_data["Mesh pipeline"].get(stage, 0) for scene_data in all_data]
        vert_values = [scene_data["Vert pipeline"].get(stage, 0) for scene_data in all_data]
        
        # Plot the Mesh bars for this stage
        ax.bar(index - bar_width / 2 - spacing / 2, mesh_values, bar_width, color=stage_colors[i], bottom=bottom_mesh)
        bottom_mesh += np.array(mesh_values)  # Update bottom for stacking

        # Plot the Vert bars for this stage
        ax.bar(index + bar_width / 2 + spacing / 2, vert_values, bar_width, color=stage_colors[i], bottom=bottom_vert)
        bottom_vert += np.array(vert_values)  # Update bottom for stacking

    # Customize the plot
    ax.set_xlabel("Scene")
    ax.set_ylabel("Cumulative VK Time (microseconds)")
    ax.set_title("Mesh pipeline vs. Vertex pipeline")
    
    # Rotate the x-axis labels to prevent overlap
    ax.set_xticks(index)
    ax.set_xticklabels([f"{scene}\n(Mesh, Vert)" for scene in x_labels], rotation=45, ha="right")
    
    
    # Create custom legend handles with the same colors for stages
    legend_handles = [mpatches.Patch(color=stage_colors[i], label=stage) for i, stage in enumerate(stages)]
    
    # Set the legend with the custom handles
    ax.legend(handles=legend_handles, title="Stages", loc='upper right')

    # Adjust layout to minimize white space
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Histogram saved as {filename}")


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

def plot_cumulative_histogram_format(
    benchmarks, 
    pipelines, 
    pipeline_names,  
    stages=["GPU Dist", "GPU Sort", "Rendering"], 
    filename="histogram.png"
):
    scene_groups = defaultdict(list)

    # Group the results by scene and pipeline
    for scene_name, _, benchmark_name, timers in benchmarks:
        if benchmark_name in pipelines and isinstance(timers, dict):
            scene_groups[scene_name].append((benchmark_name, timers))
    
    if not scene_groups:
        print("No relevant benchmarks found.")
        return
    
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
    stage_colors = [nvidia_colors["black"], nvidia_colors["dark_green"], nvidia_colors["green"]]
    
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
    ax.set_xlabel("Scene")
    ax.set_ylabel("Cumulative VK Time (microseconds)")
    ax.set_title("Pipeline Performance Comparison")

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


if __name__ == "__main__":    
    executable = os.path.abspath("bin_x64/Release/vk_gaussian_splatting.exe")
    benchmark_file = os.path.abspath("benchmark.txt")
    # Setup argument parsing for the base dataset path
    parser = argparse.ArgumentParser(description="Run benchmarks for 3D scenes.")
    parser.add_argument("dataset_path", type=str, help="Base path to the dataset")
    
    args = parser.parse_args()
    
    # Define the scenes with the relative paths
    scenes = {
        "bicycle 30000": "bicycle/bicycle/point_cloud/iteration_30000/point_cloud.ply",
        "bonsai 30000": "bonsai/bonsai/point_cloud/iteration_30000/point_cloud.ply",
        "counter 30000": "counter/point_cloud/iteration_30000/point_cloud.ply",
        "drjohnson 30000": "drjohnson/point_cloud/iteration_30000/point_cloud.ply",
        "flowers 30000": "flowers/point_cloud/iteration_30000/point_cloud.ply",
        "garden 30000": "garden/point_cloud/iteration_30000/point_cloud.ply",
        "kitchen 30000": "kitchen/point_cloud/iteration_30000/point_cloud.ply",
        "playroom 30000": "playroom/point_cloud/iteration_30000/point_cloud.ply",
        "room 30000": "room/point_cloud/iteration_30000/point_cloud.ply",
        "stump 30000": "stump/point_cloud/iteration_30000/point_cloud.ply",
        "train 30000": "train/point_cloud/iteration_30000/point_cloud.ply",
        "treehill 30000": "treehill/point_cloud/iteration_30000/point_cloud.ply",
        "truck 30000": "truck/point_cloud/iteration_30000/point_cloud.ply"
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
    # plot_cumulative_histogram(all_results)
    pipelines = ["Mesh pipeline", "Vert pipeline"]
    pipeline_names = ["Mesh", "Vert"]
    plot_cumulative_histogram_format(all_results, pipelines, pipeline_names, filename="histogram_shader.png")
    pipelines = ["Mesh pipeline", "Mesh pipeline fp16", "Mesh pipeline uint8"]
    pipeline_names = ["fp32", "fp16", "uint8"]
    plot_cumulative_histogram_format(all_results, pipelines, pipeline_names, filename="histogram_format.png")

    print("CSV and histogram generation complete.")
