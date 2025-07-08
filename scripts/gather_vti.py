import pyvista as pv
import numpy as np
import xarray as xr
from pathlib import Path
import xml.etree.ElementTree as ET


def gather_vti_files(path: Path, output: Path = Path("gathered_data.nc"), input_filename = None):
    files = sorted(path.glob("*.vti"), key=lambda f: int(f.stem.split(".")[0]))

    if not files:
        raise ValueError("No VTI files found in the specified directory.")
    first_file = files[0]
    tree = ET.parse(first_file)
    root = tree.getroot()
    # Extract metadata from the root element
    a_arrays = []
    b_arrays = []

    # Extract RD parameters
    rd_element = root.find(".//RD")
    if rd_element is not None:
        rule_element = rd_element.find(".//rule")
        if rule_element is not None:
            timestep_element = rule_element.find(".//param[@name='timestep']")
            D_a_element = rule_element.find(".//param[@name='D_a']")
            D_b_element = rule_element.find(".//param[@name='D_b']")
            alpha_element = rule_element.find(".//param[@name='alpha']")
            beta_element = rule_element.find(".//param[@name='beta']")
            dx_element = rule_element.find(".//param[@name='dx']")
            timestep = float(timestep_element.text) if timestep_element is not None else None
            D_a = float(D_a_element.text) if D_a_element is not None else None
            D_b = float(D_b_element.text) if D_b_element is not None else None
            alpha = float(alpha_element.text) if alpha_element is not None else None
            beta = float(beta_element.text) if beta_element is not None else None
            dx = float(dx_element.text) if dx_element is not None else None
            print(f"Timestep: {timestep}, D_a: {D_a}, D_b: {D_b}, Alpha: {alpha}, Beta: {beta}")
        else:
            print("Rule element not found.")
    else:
        print("RD element not found.")

    for file in files:
        grid = pv.read(file)

        dims = grid.dimensions
        a = grid["a"].reshape(dims[::-1])  # (z, y, x)
        b = grid["b"].reshape(dims[::-1])
        a_arrays.append(a[0])
        b_arrays.append(b[0])

    a_array = np.stack(a_arrays) # (n_snapshots, n_y, n_x)
    b_array = np.stack(b_arrays) # (n_snapshots, n_y, n_x)
    res = np.stack([a_array, b_array], axis=1)  # (n_snapshots, 2, n_y, n_x)
    if np.any(np.isnan(res)):
        raise ValueError("Contains NaN!. Skipping")
        
    res = np.transpose(res, (0, 1, 3, 2))  # (n_snapshots, 2, n_x, n_y)
    res = res[np.newaxis, ...]  
    res = xr.Dataset(
        data_vars={
            "data": (["trajectory", "snapshot", "component", "x", "y"], res),
            "Du": (["trajectory"], [D_a]),
            "Dv": (["trajectory"], [D_b]),
            "A": (["trajectory"], [alpha]),
            "B": (["trajectory"], [beta]),
            "input_filename": (
                ["trajectory"],
                [input_filename if input_filename else ""],
            ),
        },
        coords={
            "trajectory": [0],
            "snapshot": np.arange(len(files)),
            "component": ["u", "v"],
            "x": np.arange(dims[0]),
            "y": np.arange(dims[1]),
        },
        attrs={
            "timestep": timestep,
            "n_snapshots": len(files),
            "n_x": dims[0],
            "n_y": dims[1],
            "dx": dx,
        },
    )
    res.to_netcdf(output, mode="w", format="NETCDF4")
    print(f"Gathered {len(files)} VTI files into {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gather VTI files into a single dataset.")
    parser.add_argument("path", type=Path, help="Path to the directory containing VTI files")
    parser.add_argument("--input-filename", type=str, help="Name of the input VTI file")
    parser.add_argument("--output", "-o", type=Path, help="Path to the output netCDF file", default=Path("gathered_data.nc"))
    args = parser.parse_args()

    gather_vti_files(args.path, args.output, args.input_filename)
