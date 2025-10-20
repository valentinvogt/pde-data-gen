import numpy as np
import os
from create_netcdf_input import create_input_file
from helpers import f_scalings, zero_func, const_sigma

A = 5
B = 9


def initial_sparse_sources(member, coupled_idx, x_position, y_position):
    np.random.seed(1)

    if coupled_idx == 0:
        u = np.ones(x_position.shape)
    elif coupled_idx == 1:
        u = np.zeros(x_position.shape)
        for i in range(0, x_position.shape[0]):
            i = np.random.randint(0, x_position.shape[0])
            j = np.random.randint(0, x_position.shape[1])
            u[i, j] = 1.0
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = 0.0 * x_position
    
    return u


def steady_state_plus_noise(member, coupled_idx, x_position, y_position):
    np.random.seed(1)
    sigma = 0.1
    if coupled_idx == 0:
        u = A * np.ones(shape=x_position.shape) + np.random.normal(
            0.0, sigma, size=x_position.shape
        )
    elif coupled_idx == 1:
        u = (B / A) * np.ones(shape=x_position.shape) + np.random.normal(
            0.0, sigma, size=x_position.shape
        )
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = 0.0 * x_position
    
    # u += (
    #     0.5
    #     * np.sin(2 * np.pi * x_position / 100)
    #     * np.sin(2 * np.pi * y_position / 100)
    # )
    
    return u


def wrap(model, Nx, Nt, dt, init=None, dx=1.0):
    fn_order = 4 if model == "fhn" else 3
    fn_scalings = f_scalings(model, A, B)
    input_filename = f"data/{model}.nc"
    create_input_file(
        f"data/{model}.nc",
        f"data/{model}_output.nc",
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
        number_snapshots=20,
        n_members=1,
        initial_value_function=steady_state_plus_noise,
        sigma_function=const_sigma,
        bc_neumann_function=zero_func,
        f_value_function=fn_scalings,
        Du=2.0,
        Dv=22.0,
    )

    return input_filename

"""
model can be one of bruss, gray_scott, fhn
"""
input_filename = wrap(model="bruss", Nx=200, Nt=2_500, dt=0.0025)

# os.system(f"build/run_from_netcdf {input_filename}")