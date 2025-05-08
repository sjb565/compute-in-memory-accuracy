import os
import numpy as np
import matplotlib.pyplot as plt

# Root directory containing all activation result folders
root_dir = "."  # or wherever your directories like "activations_ibits_8_wbits_2" are located

# Scan all subdirectories that match the expected naming pattern
data = []
for folder in os.listdir(root_dir):
    if folder.startswith("activations_ibits_") and os.path.isdir(os.path.join(root_dir, folder)):
        try:
            parts = folder.split("_")
            input_bits = int(parts[2])
            weight_bits = int(parts[4])
            acc_path = os.path.join(root_dir, folder, "accuracy.npy")
            if os.path.exists(acc_path):
                accuracy = np.load(acc_path).item()  # Load float
                data.append((input_bits, weight_bits, accuracy))
        except Exception as e:
            print(f"Skipping folder {folder}: {e}")

# Convert to NumPy array for easier handling
data = np.array(data)  # shape: (N, 3)

# Sort by input_bits then weight_bits
data = data[np.lexsort((data[:,1], data[:,0]))]

# Plot using a heatmap
input_bit_vals = sorted(set(data[:,0]))
weight_bit_vals = sorted(set(data[:,1]))
acc_matrix = np.full((len(weight_bit_vals), len(input_bit_vals)), np.nan)

# Fill the accuracy matrix
for input_bits, weight_bits, acc in data:
    i = weight_bit_vals.index(weight_bits)
    j = input_bit_vals.index(input_bits)
    acc_matrix[i, j] = acc * 100  # Convert to %

# Plot
plt.figure(figsize=(8, 6))
im = plt.imshow(acc_matrix, origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(im, label='Accuracy (%)')
plt.xticks(ticks=range(len(input_bit_vals)), labels=input_bit_vals)
plt.yticks(ticks=range(len(weight_bit_vals)), labels=weight_bit_vals)
plt.xlabel("Input Bits")
plt.ylabel("Weight Bits")
plt.title("Top-1 Accuracy vs Input & Weight Bits")
plt.tight_layout()
plt.show()