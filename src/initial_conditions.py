from dataclasses import dataclass
from typing import Union, List, Callable, Literal, Dict
import numpy as np
from functools import partial
from pydantic import BaseModel, Field


class InitialCondition(BaseModel):
    pass


class NormalIC(InitialCondition):
    sigma_u: float
    sigma_v: float


class PointSourcesIC(InitialCondition):
    density: float


class UniformIC(InitialCondition):
    u_min: float
    u_max: float
    v_min: float
    v_max: float


class HexPatternIC(InitialCondition):
    amplitude: float
    wavelength: float


def ic_from_dict(d: Dict) -> InitialCondition:
    type = d["type"]
    d.pop("type")
    if type == "normal":
        return NormalIC(**d)
    elif type == "point_sources":
        return PointSourcesIC(**d)
    elif type == "uniform":
        return UniformIC(**d)
    elif type == "hex_pattern":
        return HexPatternIC(**d)
    else:
        raise ValueError(f"Unknown initial condition type: {type}")


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

    if isinstance(ic_params, NormalIC):
        u += rng.normal(0.0, ic_params.sigma_u)
        v += rng.normal(0.0, ic_params.sigma_v)
    elif isinstance(ic_params, UniformIC):
        u += rng.uniform(ic_params.u_min, ic_params.u_max, size=x_position.shape)
        v += rng.uniform(ic_params.v_min, ic_params.v_max, size=x_position.shape)
    elif isinstance(ic_params, HexPatternIC):
        u += (
            rng.normal(1.0, 0.5)
            * ic_params.amplitude
            * (
                np.cos(2 * np.pi * x_position / ic_params.wavelength)
                + np.sin(2 * np.pi * y_position / ic_params.wavelength)
                + np.cos(2 * np.pi * (x_position + y_position) / ic_params.wavelength)
            )
        )
        v += (
            rng.normal(1.0, 0.5)
            * ic_params.amplitude
            * (
                np.cos(2 * np.pi * x_position / ic_params.wavelength)
                + np.sin(2 * np.pi * y_position / ic_params.wavelength)
                + np.cos(2 * np.pi * (x_position + y_position) / ic_params.wavelength)
            )
        )
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = 0.0 * x_position
    return u if coupled_idx == 0 else v


def get_ic_function(
    model: str, A: float, B: float, ic_params: InitialCondition
) -> Callable:
    if isinstance(ic_params, PointSourcesIC):
        return partial(initial_sparse_sources, sparsity=ic_params.density)
    return partial(steady_state_plus_noise, params=(A, B), ic_params=ic_params)
