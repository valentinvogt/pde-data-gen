import netCDF4 as nc
import numpy as np
import os
import json
from typing import Dict, Any, List, Optional
from uuid import uuid4

class DatasetManager:
    """
    Manager for a NetCDF file that contains multiple trajectories.
    Each traj is identified by a unique traj_id.
    """
    def __init__(self, filepath: str, mode: str = 'a'):
        """
        Initialize the dataset NetCDF file.
        
        Args:
            filepath: Path to the NetCDF file
            mode: File access mode ('w' for new file, 'a' for append)
        """
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create or open the file
        if mode == 'w' or not os.path.exists(filepath):
            self.root = nc.Dataset(filepath, 'w')
            self._initialize_file()
        else:
            self.root = nc.Dataset(filepath, 'a')
            
    def _initialize_file(self):
        """Set up the initial structure"""
        self.root.createDimension('trajectory', None)  # Unlimited dimension for trajectories
        
        # Create variables for traj metadata
        self.root.createVariable('traj_id', str, ('trajectory',))
        self.root.createVariable('model', str, ('trajectory',))
        self.root.createVariable('A', 'f4', ('trajectory',))
        self.root.createVariable('B', 'f4', ('trajectory',))
        self.root.createVariable('Du', 'f4', ('trajectory',))
        self.root.createVariable('Dv', 'f4', ('trajectory',))
        self.root.createVariable('Nx', 'i4', ('trajectory',))
        self.root.createVariable('dx', 'f4', ('trajectory',))
        self.root.createVariable('Nt', 'i4', ('trajectory',))
        self.root.createVariable('dt', 'f4', ('trajectory',))
        self.root.createVariable('n_snapshots', 'i4', ('trajectory',))
        self.root.createVariable('input_file', str, ('trajectory',))
        self.root.createVariable('output_file', str, ('trajectory',))
        self.root.createVariable('random_seed', 'i4', ('trajectory',))
        self.root.createVariable('initial_condition', str, ('trajectory',))
        self.root.createVariable('original_point', str, ('trajectory',))
    
    def add_traj_metadata(self, traj_index: int, metadata: Dict[str, Any]):
        """
        Add metadata for a specific traj.
        
        Args:
            traj_index: Index for this traj
            metadata: Dictionary containing the traj metadata
        """
        self.root.variables['traj_id'][traj_index] = metadata.get('traj_id', '')
        self.root.variables['model'][traj_index] = metadata.get('model', '')
        self.root.variables['A'][traj_index] = metadata.get('A', 0)
        self.root.variables['B'][traj_index] = metadata.get('B', 0)
        self.root.variables['Du'][traj_index] = metadata.get('Du', 0)
        self.root.variables['Dv'][traj_index] = metadata.get('Dv', 0)
        self.root.variables['Nx'][traj_index] = metadata.get('Nx', 0)
        self.root.variables['dx'][traj_index] = metadata.get('dx', 0)
        self.root.variables['Nt'][traj_index] = metadata.get('Nt', 0)
        self.root.variables['dt'][traj_index] = metadata.get('dt', 0)
        self.root.variables['n_snapshots'][traj_index] = metadata.get('n_snapshots', 0)
        self.root.variables['output_file'][traj_index] = metadata.get('filename', '')
        self.root.variables['random_seed'][traj_index] = metadata.get('random_seed', 0)
        
        # Store complex objects as JSON strings
        self.root.variables['initial_condition'][traj_index] = json.dumps(metadata.get('initial_condition', {}))
        if 'original_point' in metadata:
            self.root.variables['original_point'][traj_index] = json.dumps(metadata.get('original_point', {}))
        else:
            self.root.variables['original_point'][traj_index] = '{}'

    def get_traj_count(self) -> int:
        """Get the current number of trajectories in the file"""
        return len(self.root.dimensions['trajectory'])
    
    def close(self):
        """Close the NetCDF file"""
        self.root.close()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def create_metadata_file(output_dir: str, config: Dict[str, Any]) -> str:
    """
    Create a metadata file for a set of trajectories.
    
    Args:
        output_dir: Directory where the file will be stored
        config: Configuration dictionary for the trajectories
        
    Returns:
        Path to the created file
    """
    
    # Create the consolidated metadata file
    filepath = os.path.join(output_dir, "_dataset.nc")
    
    with DatasetManager(filepath, 'w') as cnc:
        # Store the configuration in the file attributes
        for key, value in config.items():
            if isinstance(value, bool):
                cnc.root.setncattr(key, int(value))
            elif isinstance(value, (str, int, float)):
                cnc.root.setncattr(key, value)
            else:
                # Convert complex types to JSON strings
                cnc.root.setncattr(key, json.dumps(value))
    
    return filepath