import numpy as np
from numpy.linalg import norm
from scipy.fft import fft
from .features import compute_glcm_features, compute_directional_power

def compute_metrics(data, u_ss, v_ss, starting_idx, mode="old"):
    if not isinstance(data, np.ndarray):
        data = data.values
    
    if mode == "old":
        u = data[:, :, 0::2]
        v = data[:, :, 1::2]
    else:
        u = data[:, 0, :, :]
        v = data[:, 1, :, :]

    max_u = np.max(u)
    max_v = np.max(v)

    mean_dev_u = norm(u - u_ss, axis=(1, 2))
    mean_dev_v = norm(v - v_ss, axis=(1, 2))
    total_dev = mean_dev_u + mean_dev_v
    deviation = total_dev[-starting_idx:]

    du = np.diff(u, axis=1)
    dv = np.diff(v, axis=2)
    dx_norm = norm(du, axis=(1, 2)) + norm(dv, axis=(1, 2))
    last_dx = dx_norm[-starting_idx:]

    du_dt = np.gradient(u, axis=0)
    dv_dt = np.gradient(v, axis=0)
    dt_norm = norm(du_dt, axis=(1, 2)) + norm(dv_dt, axis=(1, 2))
    last_dt = dt_norm[-starting_idx:]

    u_avg = np.mean(u, axis=(1, 2))
    # if not isinstance(u_avg, np.ndarray):
    #     u_avg = u_avg.values
    fft_u = np.abs(fft(u_avg - u_ss)) / len(u_avg)
    fft_u[0] = 0

    final_u = u[-1, :, :]
    glcm_energy = compute_glcm_features(final_u)
    directional_power = compute_directional_power(final_u)

    rel_std_u = np.std(u, axis=(1, 2)) / np.mean(u, axis=(1, 2))
    rel_std_v = np.std(v, axis=(1, 2)) / np.mean(v, axis=(1, 2))
    rel_std_u_mean = np.mean(rel_std_u[-starting_idx:])
    rel_std_v_mean = np.mean(rel_std_v[-starting_idx:])

    return {
        "mean_deviation": np.mean(deviation),
        "std_deviation": np.std(deviation),
        "max_dx": np.max(last_dx),
        "mean_dx": np.mean(last_dx),
        "max_dt": np.max(last_dt),
        "mean_dt": np.mean(last_dt),
        "dominant_power": np.max(fft_u),
        "total_power": np.sum(fft_u),
        "max_u": max_u,
        "max_v": max_v,
        "rel_std_u": rel_std_u_mean,
        "rel_std_v": rel_std_v_mean,
        "dir_var": np.var(directional_power),
        "glcm_energy": glcm_energy,
        "init_sum": np.sum(u[0, :, :]) + np.sum(v[0, :, :]),
    } 