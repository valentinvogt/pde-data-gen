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
from src.create_vti_input import create_vti_input
from src.setup_helpers import (
    f_scalings,
    zero_func,
    const_sigma,
    ModelParams,
    SimParams,
    DatasetInfo,
)
from src.initial_conditions import (
    get_ic_function,
    InitialCondition,
    ic_from_dict,
    get_ic_type,
    get_ic_data_vti,
)

from src.dataset_manager import DatasetManager, create_metadata_file

print_filenames = False


def create_input_wrapper(
    model_params: ModelParams,
    sim_params: SimParams,
    initial_condition: InitialCondition,
    ds_info: DatasetInfo,
    traj_id: str,
    random_seed: int = None,
    use_vti: bool = True,
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

    input_file = traj_id
    if use_vti:
        input_file += ".vti"
    else:
        input_file += ".nc"
    input_filename = os.path.join(ds_info.output_dir, input_file)

    if random_seed is None:
        random_seed = randint(0, 2**31 - 1)

    if use_vti:
        u0, v0 = get_ic_data_vti(A, B, initial_condition, (Nx, Nx, 1), random_seed)
        create_vti_input(
            input_filename,
            model,
            Du,
            Dv,
            A,
            B,
            (Nx, Nx, 1),
            dt,
            "image_data",
            initial_a=u0,
            initial_b=v0,
            dx=dx,
        )
    else:
        output_filename = input_filename.replace(".nc", "_output.nc")
        ic_function = get_ic_function(model, A, B, initial_condition, random_seed)

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
            "ic_type": get_ic_type(initial_condition),
            "initial_condition": ic_data,
            "random_seed": random_seed,
            "n_snapshots": n_snapshots,
            "filename": output_filename,
            "dataset_id": ds_id,
            "traj_id": traj_id,
        }

        if original_point is not None:
            log_dict["original_point"] = original_point.model_dump()

        dataset_file = ds_info.file
        with DatasetManager(dataset_file, "a") as dataset:
            traj_index = dataset.get_traj_count()
            dataset.add_traj_metadata(traj_index, log_dict)
            log_dict["dataset_file"] = dataset_file
            log_dict["traj_index"] = traj_index


def sample_ball(
    model_params: ModelParams,
    sim_params: SimParams,
    dataset_info: DatasetInfo,
    initial_conditions: List[InitialCondition],
    sampling_std: ModelParams,
    num_samples_per_ball: int,
    num_samples_per_ic: int,
    use_vti: bool
):
    params = model_params.model_dump()
    A, B, Du, Dv = params.values()
    std_params = sampling_std.model_dump()
    sigma_A = std_params["A"] * A
    sigma_B = std_params["B"] * B
    sigma_Du = std_params["Du"] * Du
    sigma_Dv = std_params["Dv"] * Dv

    for _ in range(num_samples_per_ball):
        A_new = A + uniform(-sigma_A, sigma_A)
        B_new = B + uniform(-sigma_B, sigma_B)
        Du_new = Du + uniform(-sigma_Du, sigma_Du)
        Dv_new = Dv + uniform(-sigma_Dv, sigma_Dv)

        for ic in initial_conditions:
            for _ in range(num_samples_per_ic):
                create_input_wrapper(
                    ModelParams(A=A_new, B=B_new, Du=Du_new, Dv=Dv_new),
                    sim_params,
                    ic,
                    dataset_info,
                    traj_id=str(uuid4()),
                    original_point=model_params,
                    random_seed=None,
                    use_vti=use_vti,
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
        params = [
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
    else:
        raise ValueError(f"Invalid range type: {grid_mode}")

    return params


def parameters_from_df(df_path: str) -> List[Dict[str, float]]:
    df = pd.read_csv(df_path)
    cols = ["A", "B", "Du", "Dv"]
    if "seed" in df.columns:
        cols.append("seed")
    elif "random_seed" in df.columns:
        df = df.rename(columns={"random_seed": "seed"})
        cols.append("seed")
    # if "initial_condition" in df.columns:
    #     cols.append("initial_condition")
    #     cols.append("ic_type")
    # df = df[cols]
    return df.to_dict(orient="records")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    load_dotenv()
    # Extract configuration
    model = cfg.model
    dataset_id = cfg.dataset_id
    dataset_type = cfg.dataset_type
    center_definition = cfg.center_definition
    workdir_env_var = cfg.workdir_env_var
    use_vti = cfg.use_vti

    # Set up output directories
    data_dir = os.getenv(workdir_env_var, ".")
    output_dir = os.path.join(data_dir, "data", model, dataset_id)
    abs_out_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    #------ Save configuration for reference ------#
    config_to_dump = {
        "model": cfg.model,
        "dataset_id": cfg.dataset_id,
        "output_dir": abs_out_dir,
        "use_vti": cfg.use_vti,
        "dataset_type": cfg.dataset_type,
        "center_definition": cfg.center_definition,
        "sim_params": OmegaConf.to_container(cfg.sim_params, resolve=True),
        "initial_conditions": OmegaConf.to_container(
            cfg.initial_conditions, resolve=True
        ),
        "num_samples_per_ic": cfg.num_samples_per_ic,
    }

    if cfg.center_definition == "from_grid":
        config_to_dump["grid_mode"] = cfg.grid_mode
        config_to_dump["grid_params"] = OmegaConf.to_container(cfg.grid_params, resolve=True)
    elif cfg.center_definition == "from_df":
        config_to_dump["df_path"] = cfg.df_path

    if cfg.dataset_type == "ball":
        config_to_dump["sampling_std"] = OmegaConf.to_container(cfg.sampling_std, resolve=True)
        config_to_dump["num_samples_per_ball"] = cfg.num_samples_per_ball

    with open(os.path.join(output_dir, "_config.json"), "w") as f:
        json.dump(config_to_dump, f, indent=2)

    # Convert configuration to appropriate objects
    sim_params = SimParams(**cfg.sim_params)
    initial_conditions = [ic_from_dict(dict(ic)) for ic in cfg.initial_conditions]
    num_samples_per_ic = cfg.num_samples_per_ic

    # Generate parameter grid based on configuration
    if center_definition == "from_grid":
        param_grid = parameters_from_grid(cfg)
    elif center_definition == "from_df":
        if cfg.df_path is None:
            raise ValueError(
                "df_path must be specified when center_definition is from_df"
            )
        param_grid = parameters_from_df(cfg.df_path)
    else:
        raise ValueError(f"Invalid center definition: {center_definition}")

    # Create consolidation nc file (if netcdf mode)
    dataset_file = os.path.join(output_dir, "_dataset.nc")
    if not os.path.exists(dataset_file) and not use_vti:
        dataset_file = create_metadata_file(
            output_dir, OmegaConf.to_container(cfg, resolve=True)
        )
        print(f"Created dataset file: {dataset_file}")
    else:
        print(f"Appending to {dataset_file}")

    dataset_info = DatasetInfo(
        model=model,
        type=dataset_type,
        id=dataset_id,
        file=dataset_file,
        output_dir=output_dir,
    )

    if dataset_type == "ball":
        sampling_std = ModelParams(**cfg.sampling_std)
        num_samples_per_ball = cfg.num_samples_per_ball

    n = len(param_grid)
    j = 0
    for i, params in enumerate(param_grid):
        if i == j and n > 100:
            print(int(np.round(100 * j / n)), "%")
            j += np.round(0.1 * n)

        if dataset_type == "ball":
            sample_ball(
                ModelParams(**params),
                sim_params,
                dataset_info,
                initial_conditions,
                sampling_std,
                num_samples_per_ball,
                num_samples_per_ic,
                use_vti=use_vti,
            )
        elif dataset_type == "multi_ic":
            for ic in initial_conditions:
                for _ in range(num_samples_per_ic):
                    create_input_wrapper(
                        ModelParams(**params),
                        sim_params,
                        ic,
                        dataset_info,
                        traj_id=str(uuid4()),
                        random_seed=None,
                        use_vti=use_vti,
                    )
        elif dataset_type == "all_fixed":
            if "seed" in params:
                seed = params.pop("seed")
            else:
                seed = None
            ic_type = params.pop("ic_type")
            ic = ic_from_dict(params.pop("initial_condition"), ic_type)
            for _ in range(num_samples_per_ic):
                create_input_wrapper(
                    ModelParams(**params),
                    sim_params,
                    ic,
                    dataset_info,
                    traj_id=str(uuid4()),
                    random_seed=seed,
                    use_vti=use_vti,
                )
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")


if __name__ == "__main__":
    main()
    print("Successfully generated inputs")
