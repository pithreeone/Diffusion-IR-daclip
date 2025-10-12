import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

degradation_types = ["motion-blurry","hazy","low-light","noisy","rainy"]
# degradation_types = ["motion-blurry","hazy","low-light","rainy"]

dataset_train_folder = "../Warehouse/Diffusion-IR/datasets/train/"
fs_result_train_folder = "results/fs_result_ff_train"

residuals = []
labels = []

for deg_type in degradation_types:
    if deg_type == 'rainy':
        gt_folder = os.path.join(dataset_train_folder, deg_type, "RainTrainL", "GT")
        lq_folder = os.path.join(dataset_train_folder, deg_type, "RainTrainL", "LQ")
    else:
        gt_folder = os.path.join(dataset_train_folder, deg_type, "GT")
        lq_folder = os.path.join(dataset_train_folder, deg_type, "LQ")
    fs_folder = os.path.join(fs_result_train_folder, deg_type)

    # Get sorted file lists
    gt_files = sorted(os.listdir(gt_folder))
    lq_files = sorted(os.listdir(lq_folder))
    fs_files = sorted(os.listdir(fs_folder))

    count = 0
    for gt_name, lq_name, fs_name in zip(gt_files, lq_files, fs_files):
        if count >= 1000:
            break
        gt_path = os.path.join(gt_folder, gt_name)
        lq_path = os.path.join(lq_folder, lq_name)
        fs_path = os.path.join(fs_folder, fs_name)

        lq_img = cv2.imread(lq_path)
        fs_img = cv2.imread(fs_path)

        # Ensure images are same size and convert to float32
        if lq_img is None or fs_img is None or lq_img.shape != fs_img.shape:
            continue
        lq_img = lq_img.astype(np.float32) / 255.0
        fs_img = fs_img.astype(np.float32) / 255.0

        # Compute residual (FS - LQ)
        residual = fs_img - lq_img
        # residual = fs_img

        # Optionally: resize smaller to reduce dimension
        residual = cv2.resize(residual, (64, 64))

        # Flatten
        residual_flat = residual.flatten()

        residuals.append(residual_flat)
        labels.append(deg_type)

        count += 1

# Convert to numpy
residuals = np.array(residuals)

print(f"Total samples: {len(residuals)}, feature dimension: {residuals.shape[1]}")

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
residuals_2d = tsne.fit_transform(residuals)

# Map colors per degradation type
color_map = {
    "motion-blurry": "red",
    "hazy": "blue",
    "low-light": "green",
    "noisy": "orange",
    "rainy": "purple"
}

# Plot
plt.figure(figsize=(8, 6))
for deg_type in degradation_types:
    idx = [i for i, l in enumerate(labels) if l == deg_type]
    plt.scatter(residuals_2d[idx, 0], residuals_2d[idx, 1],
                label=deg_type, s=10, c=color_map[deg_type])

plt.title("t-SNE of Residual Embeddings (FS - LQ)")
plt.legend()
plt.tight_layout()
plt.savefig("t-sne_plot.png")