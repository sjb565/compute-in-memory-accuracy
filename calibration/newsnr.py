import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# === Configurable Parameters ===
reference_path = "activations/first_layer_outputs_torch.npy"
root_dir = "."  # Folder containing subfolders like activations_ibits_X_wbits_Y
num_channels_to_plot = 63  # Number of channels to visualize
plot_dB = True  # Convert SNR to dB

# === Load reference PyTorch activations ===
first_layer_outputs_torch = np.load(reference_path)

# === Storage: {(input_bits, weight_bits): [snr_ch0, snr_ch1, ...]} ===
snr_by_config = defaultdict(list)

# === Process each result folder ===
folders = [f for f in os.listdir(root_dir)
           if f.startswith("activations_ibits_") and os.path.isdir(os.path.join(root_dir, f))]

for folder in sorted(folders):
    try:
        parts = folder.split("_")
        input_bits = int(parts[2])
        weight_bits = int(parts[4])
    except Exception as e:
        print(f"[!] Skipping malformed folder name: {folder}")
        continue

    cross_sim_path = os.path.join(root_dir, folder, "first_layer_outputs.npy")
    if not os.path.exists(cross_sim_path):
        print(f"[!] Missing file in {folder}")
        continue

    first_layer_outputs_cross_sim = np.load(cross_sim_path)
    if first_layer_outputs_torch.shape != first_layer_outputs_cross_sim.shape:
        print(f"[!] Shape mismatch in {folder}. Skipping.")
        continue

    # === Compute SNR per channel ===
    snr_numerator = np.sum(first_layer_outputs_torch ** 2, axis=(0, 2, 3))
    snr_denominator = np.sum((first_layer_outputs_torch - first_layer_outputs_cross_sim) ** 2 + 1e-10, axis=(0, 2, 3))
    snr = snr_numerator / snr_denominator

    if plot_dB:
        snr = np.clip(snr, 1e-10, None)
        snr = 10 * np.log10(snr)

    snr_by_config[(input_bits, weight_bits)] = snr

# === Plotting ===
fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.get_cmap("viridis")  # Compatible across all Matplotlib versions

for ch in range(num_channels_to_plot):
    x_vals = []
    y_vals = []
    for (input_bits, weight_bits), snr_vals in sorted(snr_by_config.items()):
        if ch < len(snr_vals):
            x_vals.append(weight_bits)
            y_vals.append(snr_vals[ch])
    if x_vals:
        color = cmap(ch / max(1, num_channels_to_plot - 1))  # Normalize channel index
        ax.scatter(x_vals, y_vals, color=color, s=30)

# === Add colorbar for channels ===
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_channels_to_plot - 1))
cbar = fig.colorbar(sm, ax=ax, ticks=range(0, num_channels_to_plot))
cbar.set_label("Channel Index")

# === Labels and layout ===
ax.set_xlabel("Weight Bits")
ax.set_ylabel("SNR (dB)" if plot_dB else "SNR (Linear)")
ax.set_title(f"SNR per Channel vs Weight Bits (First {num_channels_to_plot} Channels)")
ax.grid(True)
plt.tight_layout()
plt.show()