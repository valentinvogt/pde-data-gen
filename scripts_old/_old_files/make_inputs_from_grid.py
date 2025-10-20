import numpy as np
import os
import sys
from dotenv import load_dotenv
from functools import partial
from uuid import uuid4
import pandas as pd
import argparse

from create_netcdf_input import create_input_file
from helpers import f_scalings, zero_func, const_sigma, create_json


def initial_sparse_sources(member, coupled_idx, x_position, y_position, sparsity):
    if coupled_idx == 0:
        u = np.ones(x_position.shape)
    elif coupled_idx == 1:
        u = np.zeros(x_position.shape)
        for i in range(0, int(np.floor(sparsity * x_position.shape[0]))):
            i = np.random.randint(0, x_position.shape[0])
            j = np.random.randint(0, x_position.shape[1])
            u[i, j] = 1.0
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = 0.0 * x_position

    return u


def steady_state_plus_noise(
    member, coupled_idx, x_position, y_position, params, ic_params, ic_seed=None
):
    rng = np.random.default_rng(ic_seed)
    A = params[0]
    B = params[1]
    steady_state = A if coupled_idx == 0 else B / A
    u = steady_state * np.ones(shape=x_position.shape)
    v = steady_state * np.ones(shape=x_position.shape)

    ic_type = ic_params["type"]
    if ic_type == "normal":
        u += rng.normal(0.0, ic_params["sigma_u"])
        v += rng.normal(0.0, ic_params["sigma_v"])
    elif ic_type == "uniform":
        u += rng.uniform(ic_params["u_min"], ic_params["u_max"], size=x_position.shape)
        v += rng.uniform(ic_params["v_min"], ic_params["v_max"], size=x_position.shape)
    elif ic_type == "hex_pattern":
        u += (
            rng.normal(1.0, 0.5)
            * ic_params["amplitude"]
            * (
                np.cos(2 * np.pi * x_position / ic_params["wavelength"])
                + np.sin(2 * np.pi * y_position / ic_params["wavelength"])
                + np.cos(
                    2 * np.pi * (x_position + y_position) / ic_params["wavelength"]
                )
            )
        )
        v += (
            rng.normal(1.0, 0.5)
            * ic_params["amplitude"]
            * (
                np.cos(2 * np.pi * x_position / ic_params["wavelength"])
                + np.sin(2 * np.pi * y_position / ic_params["wavelength"])
                + np.cos(
                    2 * np.pi * (x_position + y_position) / ic_params["wavelength"]
                )
            )
        )
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = 0.0 * x_position
    return u if coupled_idx == 0 else v


def run_wrapper(
    model,
    A,
    B,
    Nx,
    dx,
    Nt,
    dt,
    Du,
    Dv,
    ic_params,
    random_seed,
    n_snapshots,
    filename,
    run_id,
    original_point,
):
    fn_order = 4 if model == "fhn" else 3
    fn_scalings = f_scalings(model, A, B)
    input_filename = filename

    output_filename = filename.replace(".nc", "_output.nc")

    if model == "bruss":
        initial_condition = partial(
            steady_state_plus_noise, params=(A, B), ic_params=ic_params
        )
    elif model == "gray_scott":
        initial_condition = partial(
            initial_sparse_sources, sparsity=ic_params["density"]
        )
    else:
        initial_condition = partial(
            steady_state_plus_noise, params=(A, B), ic_params=ic_params
        )

    create_input_file(
        input_filename,
        output_filename,
        type_of_equation=2,
        x_size=Nx,
        x_length=Nx * dx,
        y_size=Nx,
        y_length=Nx * dx,
        boundary_value_type=2,
        scalar_type=0,
        n_coupled=2,
        coupled_function_order=fn_order,
        number_timesteps=Nt,
        final_time=Nt * dt,
        number_snapshots=n_snapshots,
        n_members=1,
        initial_value_function=initial_condition,
        sigma_function=const_sigma,
        bc_neumann_function=zero_func,
        f_value_function=fn_scalings,
        Du=Du,
        Dv=Dv,
    )

    # print(input_filename)

    create_json(
        {
            "model": model,
            "A": A,
            "B": B,
            "Nx": Nx,
            "dx": dx,
            "Nt": Nt,
            "dt": dt,
            "Du": Du,
            "Dv": Dv,
            "initial_condition": ic_params,
            "random_seed": random_seed,
            "n_snapshots": n_snapshots,
            "filename": output_filename,
            "run_id": run_id,
            "original_point": original_point,
        },
        filename.replace(".nc", ".json"),
    )


def sample_ball(
    model,
    A,
    B,
    Du,
    Dv,
    sampling_std,
    num_samples,
    num_samples_per_ic,
    sim_params,
    initial_conditions,
    run_id,
):
    """
    A, B, Du, Dv: parameters; center of the ball
    sampling_std: Dict[str, float]; relative standard deviation for each parameter
    num_samples: int; number of samples
    samples_per_ic: int; number of samples per initial condition
    sim_params: Dict[str, Any]; simulation parameters
    initial_conditions: List[Dict[str, Any]]; initial conditions
    run_id: str; run identifier
    """
    Nx = sim_params["Nx"]
    dx = sim_params["dx"]
    Nt = sim_params["Nt"]
    dt = sim_params["dt"]
    path = sim_params["path"]
    sigma_A = sampling_std["A"] * A
    sigma_B = sampling_std["B"] * B
    sigma_Du = sampling_std["Du"] * Du
    sigma_Dv = sampling_std["Dv"] * Dv

    for _ in range(num_samples):
        A_new = A + np.random.uniform(-sigma_A, sigma_A)
        B_new = B + np.random.uniform(-sigma_B, sigma_B)
        Du_new = Du + np.random.uniform(-sigma_Du, sigma_Du)
        Dv_new = Dv + np.random.uniform(-sigma_Dv, sigma_Dv)

        for ic in initial_conditions:
            for _ in range(num_samples_per_ic):
                run_wrapper(
                    model,
                    A_new,
                    B_new,
                    Nx,
                    dx,
                    Nt,
                    dt,
                    Du_new,
                    Dv_new,
                    ic,
                    random_seed=np.random.randint(0, 1000000),
                    n_snapshots=100,
                    filename=os.path.join(path, f"{uuid4()}.nc"),
                    run_id=run_id,
                    original_point={"A": A, "B": B, "Du": Du, "Dv": Dv},
                )


def main_bruss():
    model = "bruss"
    run_id = "test_old"
    load_dotenv()

    data_dir = os.getenv("DATA_DIR")
    path = os.path.join(data_dir, model, run_id)
    os.makedirs(path, exist_ok=True)

    initial_conditions = [
        {"type": "normal", "sigma_u": 0.1, "sigma_v": 0.1},
        {"type": "normal", "sigma_u": 0.25, "sigma_v": 0.25},
        # {"type": "uniform", "u_min": 0.0, "u_max": 0.25, "v_min": 0.0, "v_max": 0.25},
        # {"type": "hex_pattern", "amplitude": 0.25, "wavelength": 0.1},
    ]

    A_arr = [0.5, 1.0, 1.5]
    B_mul_arr = [1.25, 2, 3]
    Du_arr = [1, 3, 5]
    Dv_mul_arr = [4]

    sampling_std = {key: 0.1 for key in ["A", "B", "Du", "Dv"]}

    sim_params = {"Nx": 128, "dx": 1.0, "Nt": 60_000, "dt": 0.0025, "path": path}

    for A in A_arr:
        for B_mul in B_mul_arr:
            for Du in Du_arr:
                for Dv_mul in Dv_mul_arr:
                    B = B_mul * A
                    Dv = Dv_mul * Du

                    sample_ball(
                        model,
                        A,
                        B,
                        Du,
                        Dv,
                        sampling_std,
                        num_samples=2,
                        num_samples_per_ic=1,
                        sim_params=sim_params,
                        initial_conditions=initial_conditions,
                        run_id=run_id,
                    )


def main_gray_scott():
    model = "gray_scott"
    run_id = "ball_small"
    load_dotenv()

    data_dir = os.getenv("DATA_DIR")
    path = os.path.join(data_dir, model, run_id)
    os.makedirs(path, exist_ok=True)

    initial_conditions = [
        {"type": "point_sources", "density": 0.05},
        {"type": "point_sources", "density": 0.15},
    ]

    # A_arr = [0.035]
    # B_mul_arr = [1.0]
    # Du_arr = [0.1]
    # Dv_mul_arr = [0.3, 0.4, 0.5, 0.6]
    A_arr = [0.035, 0.036, 0.037, 0.038, 0.039]
    B_mul_arr = [1.0, 1.5, 2.0]
    Du_arr = [0.1, 0.15, 0.2, 0.25, 0.3]
    Dv_mul_arr = [0.3, 0.4, 0.5]

    sampling_std = {key: 0.05 for key in ["A", "B", "Du", "Dv"]}

    sim_params = {"Nx": 128, "dx": 1.0, "Nt": 50_000, "dt": 0.01, "path": path}

    for i, A in enumerate(A_arr):
        for B_mul in B_mul_arr:
            for Du in Du_arr:
                for Dv_mul in Dv_mul_arr:
                    B = B_mul * A
                    Dv = Dv_mul * Du

                    sample_ball(
                        model,
                        A,
                        B,
                        Du,
                        Dv,
                        sampling_std,
                        num_samples=15,
                        num_samples_per_ic=1,
                        sim_params=sim_params,
                        initial_conditions=initial_conditions,
                        run_id=run_id,
                    )

        print(f"{i + 1} / {len(A_arr)}")


def main_phase_transition():
    model = "bruss"
    run_id = "phase_transition2"
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    path = os.path.join(data_dir, model, run_id)
    os.makedirs(path, exist_ok=True)

    sim_params = {"Nx": 128, "dx": 1.0, "Nt": 60_000, "dt": 0.0025}
    Du = 2.0
    Dv = 22.0
    initial_condition = {"type": "normal", "sigma_u": 0.1, "sigma_v": 0.1}

    A_values = np.arange(0.5, 2.5, 0.1)

    B_A_values_coarse = np.arange(1.5, 2.5, 0.1)
    B_A_values_fine = np.arange(1.8, 2.2, 0.01)

    B_A_values = np.concatenate((B_A_values_coarse, B_A_values_fine))
    # remove duplicates
    B_A_values = np.unique(B_A_values)

    for A in A_values:
        for B_A in B_A_values:
            B = A * B_A

            run_wrapper(
                model=model,
                A=A,
                B=B,
                Nx=sim_params["Nx"],
                dx=sim_params["dx"],
                Nt=sim_params["Nt"],
                dt=sim_params["dt"],
                Du=Du,
                Dv=Dv,
                ic_params=initial_condition,
                random_seed=np.random.randint(0, 1000000),
                n_snapshots=100,
                filename=os.path.join(path, f"{uuid4()}.nc"),
                run_id=run_id,
                original_point=(A, B_A),
            )


def main_from_df(run_id, df_file):
    model = "bruss"
    run_id = "ball_int"
    initial_conditions = [
        {"type": "normal", "sigma_u": 0.1, "sigma_v": 0.1},
        {"type": "normal", "sigma_u": 0.25, "sigma_v": 0.25},
    ]
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    path = os.path.join(data_dir, model, run_id)
    os.makedirs(path, exist_ok=True)

    center_df = pd.read_csv("data/varied_points.csv")
    sigma = {key: 0.1 for key in center_df.columns}
    sim_params = {"Nx": 128, "dx": 1.0, "Nt": 60_000, "dt": 0.0025, "path": path}

    for i, row in center_df.iterrows():
        A = row["A"]
        B = row["B"]
        Du = row["Du"]
        Dv = row["Dv"]

        sample_ball(
            A, B, Du, Dv, sigma, 100, sim_params, initial_conditions, run_id=run_id
        )


if __name__ == "__main__":
    main_bruss()
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", default="bruss")
#     parser.add_argument("--run_id", default="")
#     parser.add_argument("--run_type", default="from_grid")
    
#     args = parser.parse_args()
#     model = args.model
#     run_id = args.run_id
#     run_type = args.run_type

#     if run_type == "from_grid":
#         if model == "bruss":
#             main_bruss()
#         elif model == "gray_scott":
#             main_gray_scott()
#         elif run_type == "phase_transition":
#             main_phase_transition()
#     elif run_type == "from_df":
#         main_from_df(run_id, "data/varied_points.csv")
#     else:
#         print("Invalid run type")
#         sys.exit(1)
        