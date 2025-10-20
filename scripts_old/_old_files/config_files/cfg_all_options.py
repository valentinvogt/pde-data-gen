import numpy as np

CONFIG = {
    "model": "bruss",
    "run_id": "faa",
    "run_type": "ball",  # "ball" or "one_trajectory"
    "center_definition": "from_grid",  # "from_grid" or "from_df"
    "grid_mode": "relative",  # "relative" or "absolute"
    "grid_config": {  # only used if center_definition is "from_grid"
        "A": list(np.linspace(0.1, 1.0, 3)),
        "B_over_A": list(np.linspace(1.5, 2.0, 3)),
        "Du": [1, 2, 3],
        "Dv_over_Du": [9],
    },
    "df_path": "",  # must be set if center_definition is "from_df"
    "sim_params": {
        "Nx": 128,
        "dx": 1.0,
        "Nt": 60_000,
        "dt": 0.0025,
        "n_snapshots": 100,
    },
    "initial_conditions": [
        {
            "type": "normal",
            "sigma_u": 0.1,
            "sigma_v": 0.1,
        },
        {
            "type": "normal",
            "sigma_u": 0.25,
            "sigma_v": 0.25,
        },
        {
            "type": "point_sources",
            "density": 0.1,
        },
        {
            "type": "uniform",
            "u_min": 0.1,
            "u_max": 0.2,
            "v_min": 0.1,
            "v_max": 0.2,
        },
        {
            "type": "hex_pattern",
            "amplitude": 0.1,
            "wavelength": 10,
        },
    ],
    "sampling_std": {
        "A": 0.1,
        "B": 0.1,
        "Du": 0.1,
        "Dv": 0.1,
    },
    "num_samples_per_point": 2,
    "num_samples_per_ic": 1,
}
