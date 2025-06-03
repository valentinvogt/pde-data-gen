import numpy as np
from scipy.fft import fft, fft2, fftshift
from skimage.feature import graycomatrix, graycoprops

def fft_features_log(img):
    f = fft2(img)
    fshift = fftshift(f)
    mag = np.abs(fshift)
    mag_log = np.log1p(mag)
    return mag_log

def compute_fft_features(frames):
    return np.array([fft_features_log(img) for img in frames])

def compute_glcm_features(final_u):
    gray_uint8 = (final_u * 255).astype(np.uint8)
    glcm = graycomatrix(
        gray_uint8,
        distances=[1, 5, 10],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    return np.mean(graycoprops(glcm, "energy"))

def compute_directional_power(final_u):
    f_transform = fftshift(np.abs(fft2(final_u)))
    power_spectrum = f_transform**2

    h, w = final_u.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[-center_y : h - center_y, -center_x : w - center_x]
    theta = np.arctan2(y, x)
    theta_bins = np.linspace(-np.pi, np.pi, 36)

    directional_power = []
    for j in range(len(theta_bins) - 1):
        mask = (theta >= theta_bins[j]) & (theta < theta_bins[j + 1])
        directional_power.append(np.sum(power_spectrum[mask]))

    return np.array(directional_power) / (np.sum(directional_power) + 1e-10) 