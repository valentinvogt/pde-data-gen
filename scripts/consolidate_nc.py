import os
import sys
import netCDF4 as nc
import argparse


def consolidate_outputs(consolidated_file_path):
    """
    Consolidate individual output files into a single netCDF file.

    Args:
        consolidated_file_path: Path to the consolidated metadata file
    """
    print(f"Consolidating outputs into {consolidated_file_path}")

    # Open the consolidated file
    with nc.Dataset(consolidated_file_path, "a") as root:
        # Check if we already have the outputs dimension and variables
        if "snapshot" not in root.dimensions:
            # First-time setup - create dimensions for data storage
            root.createDimension("snapshot", None)

            # Create variables for simulation output data
            # We'll create variables as needed when we encounter output files
            has_setup = False
        else:
            has_setup = True

        # Get the number of trajectories
        traj_count = len(root.dimensions["trajectory"])
        print(f"Found {traj_count} trajectories to process")

        # Process each traj
        for traj_idx in range(traj_count):
            # Get metadata for this traj
            traj_id = root.variables["traj_id"][traj_idx]
            output_file = root.variables["output_file"][traj_idx]

            print(f"Processing traj {traj_idx+1}/{traj_count}: {traj_id}")

            # Check if the output file exists
            if not os.path.exists(output_file):
                print(f"  Warning: Output file {output_file} not found, skipping")
                continue

            # Open the output file
            try:
                with nc.Dataset(output_file, "r") as ds:
                    # Get metadata
                    n_snapshots = ds.getncattr("number_snapshots")
                    n_x = ds.getncattr("n_x")
                    n_y = ds.getncattr("n_y")
                    n_coupled = ds.getncattr("n_coupled")

                    # Setup data variables if not already set up
                    if not has_setup:
                        # Create data variables with appropriate dimensions
                        x_dim = n_x + 2  # Include boundary nodes
                        y_dim = n_y + 2

                        # Create unlimited dimension for snapshots

                        # Create dimension variables if they don't exist
                        if "x_size_and_boundary" not in root.dimensions:
                            root.createDimension("x_size_and_boundary", x_dim)
                        if "n_coupled_and_y_size_and_boundary" not in root.dimensions:
                            root.createDimension(
                                "n_coupled_and_y_size_and_boundary", n_coupled * y_dim
                            )

                        # Create main data variable
                        root.createVariable(
                            "data",
                            "f4",
                            (
                                "trajectory",
                                "snapshot",
                                "x_size_and_boundary",
                                "n_coupled_and_y_size_and_boundary",
                            ),
                        )

                        # Create time variable for snapshots
                        if "time" not in root.variables:
                            root.createVariable("time", "f4", ("snapshot",))

                        has_setup = True

                    # Copy data from output file to consolidated file
                    # Assuming output file has data shape [member, snapshot, x, y*n_coupled]
                    # and we want to store it in consolidated as [traj, snapshot, x, y*n_coupled]
                    for snap_idx in range(n_snapshots):
                        data = ds.variables["data"][
                            0, snap_idx, :, :
                        ]  # 0 for first member

                        # Store in consolidated file
                        snapshot_idx = snap_idx  # This might need to be adjusted if we want to interleave snapshots
                        root.variables["data"][traj_idx, snapshot_idx, :, :] = data

                        # Store time data if we have it
                        if "time" in ds.variables:
                            root.variables["time"][snapshot_idx] = ds.variables["time"][
                                snap_idx
                            ]

                    print(f"  Added {n_snapshots} snapshots for traj {traj_idx}")

            except Exception as e:
                print(f"  Error processing output file {output_file}: {str(e)}")
                continue

    print("Consolidation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate individual output files into a single netCDF file.")
    parser.add_argument("consolidated_file_path", help="Path to the consolidated metadata file")
    args = parser.parse_args()

    if not os.path.exists(args.consolidated_file_path):
        print(f"Error: File {args.consolidated_file_path} does not exist")
        sys.exit(1)

    consolidate_outputs(args.consolidated_file_path)

