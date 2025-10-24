# Dataset Management Migration Guide

This guide explains how to migrate from the old dual-class system (`DatasetManager` + `Dataset`) to the new unified `TrajectoryDataset` class.

## What Changed?

### Old System (Removed)
- `DatasetManager` in [dataset_manager.py](src/dataset_manager.py) - write-only
- `Dataset` in [db_tools.py](src/db_tools.py) - read-only
- Manual CSV caching
- Manual index tracking
- String-based JSON encoding in NetCDF
- Awkward `get_data()` with row object handling

### New System
- `TrajectoryDataset` in [trajectory_dataset.py](src/trajectory_dataset.py) - unified read/write
- Automatic pandas conversion (no CSV needed)
- Automatic index management
- Clean xarray-based interface
- Native lazy loading

## Migration Examples

### Creating a Dataset

#### Old Way
```python
from src.dataset_manager import DatasetManager, create_metadata_file

# Create file
dataset_file = create_metadata_file(output_dir, config)

# Add trajectories
with DatasetManager(dataset_file, 'a') as dataset:
    traj_index = dataset.get_traj_count()
    dataset.add_traj_metadata(traj_index, metadata_dict)
```

#### New Way
```python
from src.trajectory_dataset import TrajectoryDataset

# Create and add in one go
with TrajectoryDataset.create(dataset_file, config) as ds:
    ds.add_trajectory(metadata_dict)
    # Index is handled automatically!
```

### Reading a Dataset

#### Old Way
```python
from src.db_tools import Dataset

ds = Dataset(data_dir, model, ds_id)
df = ds.df  # Loads or creates CSV cache
data = ds.get_data(row)
```

#### New Way
```python
from src.trajectory_dataset import TrajectoryDataset

ds = TrajectoryDataset.open(dataset_path)
df = ds.df  # Direct pandas conversion, no CSV needed
data = ds.get_data(row)  # Same interface!
```

### Querying Data

#### Old Way
```python
from src.db_tools import filter_df

# Manual filtering
df_filtered = filter_df(ds.df, A=0.5, model="fhn")
```

#### New Way
```python
# Built-in query method
df_filtered = ds.query(A=0.5, model="fhn")
```

### Adding Columns

#### Old Way
```python
# Confusing CSV vs NetCDF options
ds.add_column("new_metric", values, write_into_nc=False)
```

#### New Way
```python
# Always writes to NetCDF
ds.add_column("new_metric", values, description="My metric")
ds.save()
```

## API Compatibility

The new `TrajectoryDataset` maintains backward compatibility for:

- `get_data(row)` - accepts int, Series, or DataFrame
- `to_dataframe()` / `.df` property
- `get_dataset(model, ds_id)` helper function
- `expand_json_column()` helper function
- `create_metadata_file()` function

So most existing analysis code should work with minimal changes!

## Key Benefits

1. **Simpler**: One class instead of two
2. **Faster**: No CSV caching overhead, lazy loading built-in
3. **Cleaner**: xarray handles indexing and dimension management
4. **More powerful**: Native xarray selection and filtering
5. **Better metadata**: Store complex objects properly
6. **Type-safe**: Better type hints and validation

## Step-by-Step Migration

### For Dataset Creation Scripts

1. Replace import:
   ```python
   # Old
   from src.dataset_manager import DatasetManager, create_metadata_file

   # New
   from src.trajectory_dataset import TrajectoryDataset, create_metadata_file
   ```

2. Update context manager usage:
   ```python
   # Old
   with DatasetManager(dataset_file, 'a') as dataset:
       traj_index = dataset.get_traj_count()
       dataset.add_traj_metadata(traj_index, metadata)

   # New
   with TrajectoryDataset.open(dataset_file) as ds:
       ds.add_trajectory(metadata)
       # No need to track index manually!
   ```

### For Analysis Scripts

1. Replace import:
   ```python
   # Old
   from src.db_tools import Dataset, get_dataset

   # New
   from src.trajectory_dataset import TrajectoryDataset, get_dataset
   ```

2. Update instantiation (if needed):
   ```python
   # Old
   ds = Dataset(data_dir, model, ds_id)

   # New - use the helper function
   ds, output_dir = get_dataset(model, ds_id)

   # Or open directly
   ds = TrajectoryDataset.open(dataset_path)
   ```

3. Remove CSV file references - they're no longer needed!

## Breaking Changes

Minor breaking changes (easy to fix):

1. `Dataset` constructor signature changed:
   - Old: `Dataset(data_dir, model, ds_id, ds_file="_dataset.nc")`
   - New: Use `get_dataset(model, ds_id)` helper or `TrajectoryDataset.open(filepath)`

2. `.dataset` attribute is now `.ds`:
   - Old: `dataset.dataset.variables["data"]`
   - New: `ds.ds.variables["data"]` or just use `ds.get_data()`

3. `add_column()` no longer has `write_into_nc` parameter:
   - Always writes to NetCDF
   - Must call `.save()` explicitly

## Testing Your Migration

Run the test script to verify compatibility:

```bash
python src/test_trajectory_dataset.py
```

This will test:
- Creating datasets
- Adding trajectories
- Reading datasets
- Querying and filtering
- Backward compatibility with old files
