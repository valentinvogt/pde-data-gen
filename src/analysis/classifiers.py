import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from .features import compute_fft_features

def classify_frames_pca(frames, n_components=10, n_clusters=3):
    fft_images = compute_fft_features(frames)

    X_fft = fft_images.reshape(len(frames), -1)
    X_std = StandardScaler().fit_transform(X_fft)
    X_pca = PCA(n_components=n_components).fit_transform(X_std)
    return KMeans(n_clusters=n_clusters).fit_predict(X_pca)

def classify_pattern(frames, threshold_lc=0.02, threshold_const=0.05):
    """
    Classify patterns as 0 = constant, 1 = dots, 2 = connected ("labyrinth")
    """
    num_traj, height, width = frames.shape
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions (height, width) must be positive.")

    classifications = []
    total_area = float(height * width)

    for frame in frames:
        A_max = 0.0
        if frame.std() < threshold_const:
            classifications.append(0)
            continue
            
        if frame.min() == frame.max():
            binary_image = np.zeros_like(frame, dtype=bool)
        else:
            try:
                thresh = threshold_otsu(frame)
                binary_image = frame > thresh
            except ValueError:
                binary_image = np.zeros_like(frame, dtype=bool)

        if np.any(binary_image):
            labeled_image = label(binary_image, connectivity=2, background=0)
            props = regionprops(labeled_image)

            if props:
                component_areas = [prop.area for prop in props]
                A_max = np.max(component_areas)

        R_lc = A_max / total_area
        classifications.append(1 if R_lc > threshold_lc else 2)

    return classifications

def classify_temporal_behavior(df, deviation_threshold=1e-2, dt_threshold=50, osc_power_threshold=5e-2):
    if len(df) == 0:
        return None

    classifications = []
    for _, row in df.iterrows():
        mean_dev = row["mean_deviation"]
        mean_dt = row["mean_dt"]
        dom_power = row["dominant_power"]

        if mean_dev < deviation_threshold:
            category = "SS"
        elif mean_dt < dt_threshold:
            category = "DSS"
        elif dom_power > osc_power_threshold:
            category = "OSC"
        else:
            category = "INT"

        classifications.append(category)

    df["category"] = classifications
    return df 