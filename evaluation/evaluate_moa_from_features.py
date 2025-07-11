import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# --- Directory where extracted feature files (.pkl) are stored
DATA_ROOT = "/scratch/cv-course2025/group8"
FEATURE_DIR = os.path.join(DATA_ROOT, "bbbc021_features", "base_resnet")

# --- Lists to store features and corresponding labels
compound_features = []
compound_labels = []

# --- Loop through all .pkl files in the feature directory
for file in os.listdir(FEATURE_DIR):
    if not file.endswith(".pkl"):
        continue  # skip non-pickle files

    filepath = os.path.join(FEATURE_DIR, file)

    # Load each .pkl file: ((compound, concentration, moa), feature_tensor)
    with open(filepath, "rb") as f:
        (compound_info, feature) = pickle.load(f)
        compound, conc, moa = compound_info

        # Skip samples with unknown MoA
        if moa == 'null':
            continue

        # Store the numpy version of the feature tensor and its label
        compound_features.append(feature.numpy())
        compound_labels.append((compound, moa))

# --- Convert lists to structured arrays
compound_features = np.stack(compound_features)  # shape: [num_compounds, feature_dim]
compounds = [c for (c, m) in compound_labels]    # compound names
moas = [m for (c, m) in compound_labels]         # mechanism of action labels

# --- Compute cosine similarity between each compound's feature vector
sim_matrix = cosine_similarity(compound_features)  # shape: [N, N]
np.fill_diagonal(sim_matrix, -1)  # set self-similarity to -1 to exclude from top-1

# --- Top-1 NSC Accuracy:
# For each compound, check if its most similar neighbor (excluding itself)
# has the same MoA
correct = 0
for i, moa_i in enumerate(moas):
    most_sim_idx = np.argmax(sim_matrix[i])  # index of most similar other compound
    moa_j = moas[most_sim_idx]               # its MoA
    if moa_i == moa_j:
        correct += 1

# --- Compute and report top-1 not-same-compound MoA accuracy
accuracy = correct / len(compounds)
print(f"MoA Top-1 NSC Accuracy: {accuracy:.4f}")


# first run python experiments/feature_extraction/extractor.py
# then python evaluate_moa_from_features.py
