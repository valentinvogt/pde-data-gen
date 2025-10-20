import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os

# Open the NetCDF file
filename = input('Enter filename: ')

file_path = 'data/' + filename
dataset = nc.Dataset(file_path, mode='r')

# Extract dimensions
n_members = dataset.dimensions['n_members'].size
n_snapshots = dataset.dimensions['n_snapshots'].size
x_size_and_boundary = dataset.dimensions['x_size_and_boundary'].size
n_coupled_and_y_size_and_boundary = dataset.dimensions['n_coupled_and_x_size_and_boundary'].size
n_coupled = dataset.getncattr('n_coupled')
x_size = dataset.getncattr('x_length')
y_size = dataset.getncattr('y_length')

# Extract the data variable
data = dataset.variables['data'][:]

global_min = np.min(data)
global_max = np.max(data)
print(global_min)
print(global_max)

for member in range(n_members):
    for snapshot in range(n_snapshots):
        # Create a figure for each snapshot
        fig, axes = plt.subplots(nrows=n_coupled, ncols=1, figsize=(10, 10 * n_coupled),
                                 gridspec_kw={'width_ratios': [1], 'height_ratios': [1] * n_coupled, 'wspace': 0.1, 'hspace': 0.1})

        # plot only first functions
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10),
                                 # gridspec_kw={'width_ratios': [1], 'height_ratios': [1], 'wspace': 0.1, 'hspace': 0.1})
        if n_coupled == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one coupled variable


        for coupled_idx in range(n_coupled):
        # for coupled_idx in range(1):

            ax = axes[coupled_idx]

            # Extract the 2D matrix for the current member, snapshot, and coupled index
            matrix = data[member, snapshot, :, coupled_idx::n_coupled]

            # Plot the matrix
            im = ax.imshow(matrix, cmap='viridis', aspect='equal', vmin=global_min, vmax=global_max)
            ax.set_title(f'Member {member + 1}, Snapshot {snapshot + 1}, Coupled Index {coupled_idx + 1}')
            ax.set_xlabel('y')
            ax.set_ylabel('x')
            # ax.set_xlim(0, y_size)
            # ax.set_ylim(x_size, 0)
        # Add a colorbar to the figure
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Data Value')

        # Adjust layout to prevent overlap
        # plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust rect to make room for colorbar

        # Save the figure to a file
        output_filename = f'out/{os.path.splitext(filename)[0]}_member{member + 1}_snapshot{snapshot + 1}.pdf'
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

# Close the dataset
dataset.close()
