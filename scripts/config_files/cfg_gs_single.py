import numpy as np

CONFIG = {
    "model": "gray_scott",
    "run_id": "splitting",
    "run_type": "one_trajectory",  # "ball" or "one_trajectory"
    "center_definition": "from_grid",  # "from_grid" or "from_df"
    "grid_mode": "absolute",  # "relative" or "absolute"
    "grid_config": {  # only used if center_definition is "from_grid"
        "A": [0.037506],
        "B": [0.077909],
        "Du": [0.187498],
        "Dv": [0.055323],
    },
    "seed": 1,
    # "df_path": "data/pt.csv",  # must be set if center_definition is "from_df"
    "sim_params": {
        "Nx": 512,
        "dx": 1.0,
        "Nt": 500_000,
        "dt": 0.01,
        "n_snapshots": 200,
    },
    "initial_conditions": [
        # {
        #     "type": "normal",
        #     "sigma_u": 0.1,
        #     "sigma_v": 0.1,
        # },
        # {
        #     "type": "normal",
        #     "sigma_u": 0.25,
        #     "sigma_v": 0.25,
        # },
        {
            "type": "point_sources",
            "density": 0.05,
        },
        # {
        #     "type": "uniform",
        #     "u_min": 0.1,
        #     "u_max": 0.2,
        #     "v_min": 0.1,
        #     "v_max": 0.2,
        # },
        # {
        #     "type": "hex_pattern",
        #     "amplitude": 0.1,
        #     "wavelength": 10,
        # },
    ],
    # "sampling_std": {
    #     "A": 0.1,
    #     "B": 0.1,
    #     "Du": 0.1,
    #     "Dv": 0.1,
    # },
    "num_samples_per_point": 1,
    "num_samples_per_ic": 5,
}
