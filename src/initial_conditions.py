from dataclasses import dataclass
from typing import Union, List, Callable, Literal, Dict
import numpy as np
from functools import partial
from pydantic import BaseModel, Field
import json

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


def ic_from_dict(d: Dict, ic_type: str = None) -> InitialCondition:
    if type(d) is str:
        s_fixed = d.replace('""', '"')
        s_fixed = s_fixed.replace("'", '"')
        d = json.loads(s_fixed)
    if ic_type is None:
        ic_type = d.get("type", None)
        if ic_type is None:
            raise ValueError("ic_type must be passed or found in the dictionary.")
    if ic_type == "normal":
        return NormalIC(**d)
    elif ic_type == "point_sources":
        return PointSourcesIC(**d)
    elif ic_type == "uniform":
        return UniformIC(**d)
    elif ic_type == "hex_pattern":
        return HexPatternIC(**d)
    else:
        raise ValueError(f"Unknown initial condition type: {ic_type}")


def vti_initial_sparse_sources(
    dims, sparsity, ic_seed=None
):
    rng = np.random.default_rng(ic_seed)
    u = np.ones(shape=dims)
    v = np.zeros(shape=dims)
    for i in range(0, int(np.floor(sparsity * dims[0] * dims[1]))):
        i = rng.integers(0, dims[0])
        j = rng.integers(0, dims[1])
        v[i, j] = 1.0
    return u, v

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


def vti_steady_state_plus_noise(
    dims, params, ic_params, ic_seed=None
):
    rng = np.random.default_rng(ic_seed)
    A = params[0]
    B = params[1]
    u = A * np.ones(shape=dims)
    v = B * np.ones(shape=dims)
    if isinstance(ic_params, NormalIC):
        u += rng.normal(0.0, ic_params.sigma_u, size=dims)
        v += rng.normal(0.0, ic_params.sigma_v, size=dims)
    elif isinstance(ic_params, UniformIC):
        u += rng.uniform(ic_params.u_min, ic_params.u_max, size=dims)
        v += rng.uniform(ic_params.v_min, ic_params.v_max, size=dims)
    elif isinstance(ic_params, HexPatternIC):
        u += (
            rng.normal(1.0, 0.5)
            * ic_params.amplitude
            * (
                np.cos(2 * np.pi * np.arange(dims[0])[:, None] / ic_params.wavelength)
                + np.sin(2 * np.pi * np.arange(dims[1])[None, :] / ic_params.wavelength)
                + np.cos(2 * np.pi * (np.arange(dims[0])[:, None] + np.arange(dims[1])[None, :]) / ic_params.wavelength)
            )
        )
        v += (
            rng.normal(1.0, 0.5)
            * ic_params.amplitude
            * (
                np.cos(2 * np.pi * np.arange(dims[0])[:, None] / ic_params.wavelength)
                + np.sin(2 * np.pi * np.arange(dims[1])[None, :] / ic_params.wavelength)
                + np.cos(2 * np.pi * (np.arange(dims[0])[:, None] + np.arange(dims[1])[None, :]) / ic_params.wavelength)
            )
        )
    else:
        print("unknown initial condition type, returning steady state without noise")
        u = A * np.ones(shape=dims)
        v = B * np.ones(shape=dims)
    return u, v


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
        u += rng.normal(0.0, ic_params.sigma_u, size=x_position.shape)
        v += rng.normal(0.0, ic_params.sigma_v, size=x_position.shape)
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
    model: str, A: float, B: float, ic_params: InitialCondition, seed: float
) -> Callable:
    if isinstance(ic_params, PointSourcesIC):
        return partial(initial_sparse_sources, sparsity=ic_params.density)
    return partial(steady_state_plus_noise, params=(A, B), ic_params=ic_params, ic_seed=seed)


def get_ic_data_vti(
    A, B, ic_params: InitialCondition, dims, seed: float = None
):
    if isinstance(ic_params, PointSourcesIC):
        return vti_initial_sparse_sources(dims, ic_params.density, ic_seed=seed)
    else:
        return vti_steady_state_plus_noise(
            dims, params=(A, B), ic_params=ic_params, ic_seed=seed
        )
    

def get_ic_type(ic_params: InitialCondition) -> Literal["normal", "point_sources", "uniform", "hex_pattern"]:
    if isinstance(ic_params, NormalIC):
        return "normal"
    elif isinstance(ic_params, PointSourcesIC):
        return "point_sources"
    elif isinstance(ic_params, UniformIC):
        return "uniform"
    elif isinstance(ic_params, HexPatternIC):
        return "hex_pattern"
    else:
        raise ValueError(f"Unknown initial condition type: {type(ic_params)}")