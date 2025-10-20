import numpy as np
import os
import sys
import argparse
from create_netcdf_input import create_input_file
from helpers import f_scalings, zero_func, const_sigma, create_json


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


"""
Usage:
python rd_script.py model A B Nx dx Nt dt Du Dv n_snapshots

args:
model: str = "bruss", "gray_scott", "fhn"
A: float = 5
B: float = 9
Nx: int = 100
dx: float = 1.0
Nt: int = 1000
dt: float = 0.01
Du: float = 2.0
Dv: float = 22.0
sigma_ic: float = 0.1
random_seed: int = 1
n_snapshots: int = 100
filename: str = "data/bruss.nc"
run_id: str = ""
"""


def initial_sparse_sources(member, coupled_idx, x_position, y_position):
    np.random.seed(random_seed)

    if coupled_idx == 0:
        u = np.ones(x_position.shape)
    elif coupled_idx == 1:
        u = np.zeros(x_position.shape)
        for i in range(0, sparsity * x_position.shape[0]):
            i = np.random.randint(0, x_position.shape[0])
            j = np.random.randint(0, x_position.shape[1])
            u[i, j] = 1.0
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = 0.0 * x_position

    return u


def steady_state_plus_noise(member, coupled_idx, x_position, y_position):
    np.random.seed(random_seed)
    if coupled_idx == 0:
        u = A * np.ones(shape=x_position.shape) + np.random.normal(
            0.0, sigma_ic[0], size=x_position.shape
        )
    elif coupled_idx == 1:
        u = (B / A) * np.ones(shape=x_position.shape) + np.random.normal(
            0.0, sigma_ic[1], size=x_position.shape
        )
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = 0.0 * x_position

    return u


def run_wrapper(
    model,
    A, B,
    Nx, dx,
    Nt, dt,
    Du, Dv,
    ic_type, ic_param,
    random_seed,
    n_snapshots,
    filename,
    run_id,
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bruss")
    parser.add_argument("--A", type=float, default=5)
    parser.add_argument("--B", type=float, default=9)
    parser.add_argument("--Nx", type=int, default=100)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--Nt", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Du", type=float, default=2.0)
    parser.add_argument("--Dv", type=float, default=22.0)
    parser.add_argument("--sigma_ic_u", type=float, default=0.1)
    parser.add_argument("--sigma_ic_v", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--sparsity", type=int, default=1)
    parser.add_argument("--n_snapshots", type=int, default=100)
    parser.add_argument("--filename", type=str, default="data/bruss.nc")
    parser.add_argument("--run_id", type=str, default="")


    args = parser.parse_args()
    model = args.model
    A = args.A
    B = args.B
    Nx = args.Nx
    dx = args.dx
    Nt = args.Nt
    dt = args.dt
    Du = args.Du
    Dv = args.Dv
    sigma_ic_u = args.sigma_ic_u
    sigma_ic_v = args.sigma_ic_v
    if sigma_ic_v == 0.0:
        sigma_ic_v = sigma_ic_u
    sigma_ic = (sigma_ic_u, sigma_ic_v)
    random_seed = args.random_seed
    sparsity = args.sparsity
    n_snapshots = args.n_snapshots
    filename = args.filename
    run_id = args.run_id


    fn_order = 4 if model == "fhn" else 3
    fn_scalings = f_scalings(model, A, B)
    input_filename = filename

    output_filename = filename.replace(".nc", "_output.nc")

    if model == "bruss":
        initial_condition = steady_state_plus_noise
    elif model == "gray_scott":
        initial_condition = initial_sparse_sources
    else:
        initial_condition = steady_state_plus_noise

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

    print(input_filename)

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
            "sigma_ic_u": sigma_ic_u,
            "sigma_ic_v": sigma_ic_v,
            "random_seed": random_seed,
            "n_snapshots": n_snapshots,
            "filename": output_filename,
            "run_id": run_id,
        },
        filename.replace(".nc", ".json"),
    )
