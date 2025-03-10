import numpy as np
from numpy.random import uniform, randint
import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
from uuid import uuid4
import pandas as pd
from typing import List, Dict
from itertools import product
from src.create_netcdf_input import create_input_file
from src.setup_helpers import (
    f_scalings,
    zero_func,
    const_sigma,
    create_json,
    ModelParams,
    SimParams,
    DatasetInfo,
)
from src.initial_conditions import (
    get_ic_function,
    InitialCondition,
    ic_from_dict,
)

from src.dataset_manager import DatasetManager, create_metadata_file

print_filenames = False


def run_wrapper(
    model_params: ModelParams,
    sim_params: SimParams,
    initial_condition: InitialCondition,
    ds_info: DatasetInfo,
    run_id: str,
    random_seed: int = None,
    original_point: ModelParams = None,
):
    model = ds_info.model
    ds_id = ds_info.id

    params = model_params.model_dump()
    A, B, Du, Dv = params.values()

    sim_values = sim_params.model_dump()
    Nx, dx, Nt, dt, n_snapshots = sim_values.values()

    ic_data = initial_condition.model_dump(mode="json")

    fn_order = 4 if model == "fhn" else 3
    fn_scalings = f_scalings(model, A, B)

    input_file = run_id + ".nc"
    input_filename = os.path.join(ds_info.output_dir, input_file)
    output_filename = input_filename.replace(".nc", "_output.nc")

    ic_function = get_ic_function(model, A, B, initial_condition)

    if random_seed is None:
        random_seed = randint(0, 2**32 - 1)

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
        initial_value_function=ic_function,
        sigma_function=const_sigma,
        bc_neumann_function=zero_func,
        f_value_function=fn_scalings,
        Du=Du,
        Dv=Dv,
    )

    log_dict = {
        "model": model,
        "A": A,
        "B": B,
        "Nx": Nx,
        "dx": dx,
        "Nt": Nt,
        "dt": dt,
        "Du": Du,
        "Dv": Dv,
        "initial_condition": ic_data,
        "random_seed": random_seed,
        "n_snapshots": n_snapshots,
        "filename": output_filename,
        "dataset_id": ds_id,
        "run_id": run_id,
    }

    if original_point is not None:
        log_dict["original_point"] = original_point.model_dump()

    dataset_file = ds_info.file
    with DatasetManager(dataset_file, "a") as dataset:
        run_index = dataset.get_run_count()
        dataset.add_run_metadata(run_index, log_dict)
        log_dict["dataset_file"] = dataset_file
        log_dict["run_index"] = run_index

    # backward compatibility
    create_json(
        log_dict,
        input_filename.replace(".nc", ".json"),
    )

    if print_filenames:
        print(input_filename)


def sample_ball(
    model_params: ModelParams,
    sim_params: SimParams,
    dataset_info: DatasetInfo,
    path: str,
    initial_conditions: List[InitialCondition],
    sampling_std: ModelParams,
    num_samples_per_point: int,
    num_samples_per_ic: int,
):
    params = model_params.model_dump()
    A, B, Du, Dv = params.values()
    std_params = sampling_std.model_dump()
    sigma_A = std_params["A"] * A
    sigma_B = std_params["B"] * B
    sigma_Du = std_params["Du"] * Du
    sigma_Dv = std_params["Dv"] * Dv

    for _ in range(num_samples_per_point):
        A_new = A + uniform(-sigma_A, sigma_A)
        B_new = B + uniform(-sigma_B, sigma_B)
        Du_new = Du + uniform(-sigma_Du, sigma_Du)
        Dv_new = Dv + uniform(-sigma_Dv, sigma_Dv)

        for ic in initial_conditions:
            for _ in range(num_samples_per_ic):
                run_wrapper(
                    ModelParams(A=A_new, B=B_new, Du=Du_new, Dv=Dv_new),
                    sim_params,
                    ic,
                    dataset_info,
                    run_id=str(uuid4()),
                    original_point=model_params,
                )


def ball_sampling(
    centers: List[ModelParams],
    sim_params: SimParams,
    dataset_info: DatasetInfo,
    initial_conditions: List[InitialCondition],
    sampling_std: ModelParams,
    num_samples_per_point: int,
    num_samples_per_ic: int,
):
    data_dir = os.getenv("DATA_DIR")
    output_dir = dataset_info.output_dir
    path = os.path.join(data_dir, output_dir)
    os.makedirs(path, exist_ok=True)

    j = 0
    n = len(centers)
    for i, center in enumerate(centers):
        if i == j:
            print(int(np.round(100 * j / n)), "%")
            j += np.round(0.1 * n)
        sample_ball(
            center,
            sim_params,
            dataset_info,
            path,
            initial_conditions,
            sampling_std,
            num_samples_per_point,
            num_samples_per_ic,
        )


def parameters_from_grid(cfg: DictConfig) -> List[Dict[str, float]]:
    grid_mode = cfg.grid_mode
    grid_params = cfg.grid_params

    if grid_mode == "absolute":
        param_ranges = [
            grid_params.A,
            grid_params.B,
            grid_params.Du,
            grid_params.Dv,
        ]
        return [
            {"A": A, "B": B, "Du": Du, "Dv": Dv}
            for A, B, Du, Dv in product(*param_ranges)
        ]
    elif grid_mode == "relative":
        params = []
        for A in grid_params.A:
            for B_over_A in grid_params.B_over_A:
                for Du in grid_params.Du:
                    for Dv_over_Du in grid_params.Dv_over_Du:
                        B = A * B_over_A
                        Dv = Du * Dv_over_Du
                        params.append({"A": A, "B": B, "Du": Du, "Dv": Dv})
        return params
    else:
        raise ValueError(f"Invalid range type: {grid_mode}")


def parameters_from_df(df_path: str) -> pd.DataFrame:
    df = pd.read_csv(df_path)
    df = df[["A", "B", "Du", "Dv"]]
    return df.to_dict(orient="records")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    load_dotenv()
    
    # Extract configuration
    model = cfg.model
    dataset_id = cfg.dataset_id
    dataset_type = cfg.dataset_type
    center_definition = cfg.center_definition
    print(f"Creating a {model} dataset with ID {dataset_id}")
    
    # Set up output directories
    if cfg.location == "work":
        data_dir = os.getenv("WORK_DIR")
    else:
        data_dir = os.getenv("SCRATCH_DIR")
    output_dir = os.path.join(data_dir, "data", model, dataset_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration for reference
    with open(os.path.join(output_dir, "_config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
    
    # Convert configuration to appropriate objects
    sim_params = SimParams(**cfg.sim_params)
    initial_conditions = [ic_from_dict(dict(ic)) for ic in cfg.initial_conditions]
    
    # Generate parameter grid based on configuration
    if center_definition == "from_grid":
        param_grid = parameters_from_grid(cfg)
    elif center_definition == "from_df":
        if cfg.df_path is None:
            raise ValueError("df_path must be specified when center_definition is from_df")
        param_grid = parameters_from_df(cfg.df_path)
    else:
        raise ValueError(f"Invalid center definition: {center_definition}")
    
    # Create metadata for this dataset
    dataset_file = create_metadata_file(output_dir, OmegaConf.to_container(cfg, resolve=True))
    print(f"Created dataset file: {dataset_file}")
    
    dataset_info = DatasetInfo(
        model=model,
        type=dataset_type,
        id=dataset_id,
        file=dataset_file,
        output_dir=output_dir,
    )
    
    # dataset sampling strategy based on configuration
    if dataset_type == "ball":
        sampling_std = ModelParams(**cfg.sampling_std)
        centers = [ModelParams(**center) for center in param_grid]
        num_samples_per_point = cfg.num_samples_per_point
        num_samples_per_ic = cfg.num_samples_per_ic
        
        ball_sampling(
            centers,
            sim_params,
            dataset_info,
            initial_conditions,
            sampling_std,
            num_samples_per_point,
            num_samples_per_ic,
        )
    elif dataset_type == "one_trajectory":
        n = len(param_grid)
        j = 0
        for i, center in enumerate(param_grid):
            if i == j:
                print(int(np.round(100 * j / n)), "%")
                j += np.round(0.1 * n)
            for ic in initial_conditions:
                run_wrapper(
                    ModelParams(**center),
                    sim_params,
                    ic,
                    dataset_info,
                    run_id=str(uuid4()),
                )
    else:
        raise ValueError(f"Invalid run_type: {dataset_type}")


if __name__ == "__main__":
    main()
