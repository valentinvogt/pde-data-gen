import netCDF4 as nc
import numpy as np
import os
import json
from typing import Dict, Any, List, Optional
from uuid import uuid4

class DatasetManager:
    """
    Manager for a NetCDF file that contains multiple runs.
    Each run is identified by a unique run_id.
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
        self.root.createDimension('run', None)  # Unlimited dimension for runs
        
        # Create variables for run metadata
        self.root.createVariable('run_id', str, ('run',))
        self.root.createVariable('model', str, ('run',))
        self.root.createVariable('A', 'f4', ('run',))
        self.root.createVariable('B', 'f4', ('run',))
        self.root.createVariable('Du', 'f4', ('run',))
        self.root.createVariable('Dv', 'f4', ('run',))
        self.root.createVariable('Nx', 'i4', ('run',))
        self.root.createVariable('dx', 'f4', ('run',))
        self.root.createVariable('Nt', 'i4', ('run',))
        self.root.createVariable('dt', 'f4', ('run',))
        self.root.createVariable('n_snapshots', 'i4', ('run',))
        self.root.createVariable('input_file', str, ('run',))
        self.root.createVariable('output_file', str, ('run',))
        self.root.createVariable('random_seed', 'i4', ('run',))
        self.root.createVariable('initial_condition', str, ('run',))
        self.root.createVariable('original_point', str, ('run',))
    
    def add_run_metadata(self, run_index: int, metadata: Dict[str, Any]):
        """
        Add metadata for a specific run.
        
        Args:
            run_index: Index for this run
            metadata: Dictionary containing the run metadata
        """
        self.root.variables['run_id'][run_index] = metadata.get('run_id', '')
        self.root.variables['model'][run_index] = metadata.get('model', '')
        self.root.variables['A'][run_index] = metadata.get('A', 0)
        self.root.variables['B'][run_index] = metadata.get('B', 0)
        self.root.variables['Du'][run_index] = metadata.get('Du', 0)
        self.root.variables['Dv'][run_index] = metadata.get('Dv', 0)
        self.root.variables['Nx'][run_index] = metadata.get('Nx', 0)
        self.root.variables['dx'][run_index] = metadata.get('dx', 0)
        self.root.variables['Nt'][run_index] = metadata.get('Nt', 0)
        self.root.variables['dt'][run_index] = metadata.get('dt', 0)
        self.root.variables['n_snapshots'][run_index] = metadata.get('n_snapshots', 0)
        self.root.variables['output_file'][run_index] = metadata.get('filename', '')
        self.root.variables['random_seed'][run_index] = metadata.get('random_seed', 0)
        
        # Store complex objects as JSON strings
        self.root.variables['initial_condition'][run_index] = json.dumps(metadata.get('initial_condition', {}))
        if 'original_point' in metadata:
            self.root.variables['original_point'][run_index] = json.dumps(metadata.get('original_point', {}))
        else:
            self.root.variables['original_point'][run_index] = '{}'
    
    def get_run_count(self) -> int:
        """Get the current number of runs in the file"""
        return len(self.root.dimensions['run'])
    
    def close(self):
        """Close the NetCDF file"""
        self.root.close()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def create_metadata_file(output_dir: str, config: Dict[str, Any]) -> str:
    """
    Create a metadata file for a set of runs.
    
    Args:
        output_dir: Directory where the file will be stored
        config: Configuration dictionary for the runs
        
    Returns:
        Path to the created file
    """
    
    # Create the consolidated metadata file
    filepath = os.path.join(output_dir, "_dataset.nc")
    
    with DatasetManager(filepath, 'w') as cnc:
        # Store the configuration in the file attributes
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                cnc.root.setncattr(key, value)
            else:
                # Convert complex types to JSON strings
                cnc.root.setncattr(key, json.dumps(value))
    
    return filepath