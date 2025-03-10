import os
import sys
import json
import netCDF4 as nc
import numpy as np
from glob import glob
from src.dataset_manager import DatasetManager

def consolidate_old_format(input_dir, output_file=None):
    """
    Consolidate files from an old dataset format into a new dataset file.
    
    Args:
        input_dir: Directory containing .json files and their corresponding .nc files
        output_file: Path to the output consolidated file. If None, creates 
                     _dataset.nc in the input_dir.
    """
    # Find all JSON files in the input directory
    json_files = glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    # Create output file if not specified
    if output_file is None:
        output_file = os.path.join(input_dir, "_dataset.nc")
    
    # Create the consolidated metadata file
    print(f"Creating consolidated file: {output_file}")
    
    # First, read one JSON to get model information for the config
    with open(json_files[0], 'r') as f:
        sample_json = json.load(f)
        
    # Create basic config from sample
    config = {
        "model": sample_json.get("model", ""),
        "dataset_id": sample_json.get("run_id", "old_dataset"),
        "dataset_type": "consolidated"
    }
    
    # Create initial dataset
    with DatasetManager(output_file, 'w') as dataset:
        # Store the configuration in the file attributes
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                dataset.root.setncattr(key, value)
            else:
                # Convert complex types to JSON strings
                dataset.root.setncattr(key, json.dumps(value))
    
    # Process each JSON file
    run_index = 0
    valid_outputs = []
    
    for json_path in json_files:
        # Load the JSON data
        if os.path.basename(json_path).startswith("_"):
            continue
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        # Verify output file exists
        output_file_path = metadata.get("filename", "")
        output_file_path = output_file_path.replace("/cluster/scratch/vogtva", "/cluster/work/math/vogtva")
        if not os.path.exists(output_file_path):
            print(f"Warning: Output file {output_file_path} not found, skipping")
            continue
        
        print(f"Processing run {run_index+1}/{len(json_files)}: {os.path.basename(json_path)}")
        
        # Add to dataset
        with DatasetManager(output_file, 'a') as dataset:
            dataset.add_run_metadata(run_index, metadata)
            valid_outputs.append(metadata)
            run_index += 1
    
    # Consolidate the output files
    consolidate_outputs(output_file, valid_outputs)
    print(f"Successfully consolidated {run_index} runs into {output_file}")


def consolidate_outputs(consolidated_file_path, valid_outputs):
    """
    Consolidate individual output files into a single netCDF file.

    Args:
        consolidated_file_path: Path to the consolidated metadata file
        valid_outputs: List of valid metadata entries with existing output files
    """
    print(f"Consolidating outputs into {consolidated_file_path}")

    # Open the consolidated file
    with nc.Dataset(consolidated_file_path, "a") as root:
        # Create dimensions for data storage
        root.createDimension("snapshot", None)
        
        # Process the first output file to get dimensions
        first_output = valid_outputs[0]["filename"]
        with nc.Dataset(first_output, "r") as ds:
            # Get metadata
            n_snapshots = ds.getncattr("number_snapshots")
            n_x = ds.getncattr("n_x")
            n_y = ds.getncattr("n_y")
            n_coupled = ds.getncattr("n_coupled")
            
            # Create data variables with appropriate dimensions
            x_dim = n_x + 2  # Include boundary nodes
            y_dim = n_y + 2
            
            # Create dimension variables
            root.createDimension("x_size_and_boundary", x_dim)
            root.createDimension("n_coupled_and_y_size_and_boundary", n_coupled * y_dim)
            
            # Create main data variable
            root.createVariable(
                "data",
                "f4",
                (
                    "run",
                    "snapshot",
                    "x_size_and_boundary",
                    "n_coupled_and_y_size_and_boundary",
                ),
            )
            
            # Create time variable for snapshots
            root.createVariable("time", "f4", ("snapshot",))
        
        # Process each run
        for run_idx, metadata in enumerate(valid_outputs):
            output_file = metadata["filename"]
            
            print(f"Processing run {run_idx+1}/{len(valid_outputs)}: {os.path.basename(output_file)}")
            
            # Open the output file
            try:
                with nc.Dataset(output_file, "r") as ds:
                    # Get metadata
                    n_snapshots = ds.getncattr("number_snapshots")
                    
                    # Copy data from output file to consolidated file
                    for snap_idx in range(n_snapshots):
                        data = ds.variables["data"][0, snap_idx, :, :]  # 0 for first member
                        
                        # Store in consolidated file
                        root.variables["data"][run_idx, snap_idx, :, :] = data
                        
                        # Store time data if we have it
                        if "time" in ds.variables:
                            root.variables["time"][snap_idx] = ds.variables["time"][snap_idx]
                
                print(f"  Added {n_snapshots} snapshots for run {run_idx}")
                
            except Exception as e:
                print(f"  Error processing output file {output_file}: {str(e)}")
                continue

    print("Consolidation complete")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python consolidate_old_format.py <input_directory> [output_file]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.isdir(input_dir):
        print(f"Error: Directory {input_dir} does not exist")
        sys.exit(1)
    
    consolidate_old_format(input_dir, output_file)