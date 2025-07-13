import torch
import numpy as np
from sklearn.decomposition import PCA

class TypicalVariationNormalizer:
    """
    Implements Typical Variation Normalization (TVN) as a preprocessing step for feature vectors.

    TVN is commonly used in bioimage analysis to reduce batch effects and normalize 
    features relative to a control group (typically DMSO-treated images). It uses PCA 
    to capture dominant variation modes and whitens the features.

    Attributes:
        mean_ (torch.Tensor): Mean vector of the DMSO features computed during PCA fitting.
        components_ (torch.Tensor): Principal components (eigenvectors) from PCA.
        whiten (bool): If True, uses PCA whitening during fitting.
        eps (float): Small epsilon value to ensure numerical stability.
    """
    def __init__(self, eps=1e-8):
        self.mean_ = None
        self.components_ = None
        self.whiten = True
        self.eps = eps

    def fit(self, dmso_features: torch.Tensor):
        """
        Fits the PCA model on the given DMSO feature set, storing the mean and components.
        """
        # Convert to numpy for sklearn PCA
        pca = PCA(n_components=dmso_features.shape[1], whiten=self.whiten)
        pca.fit(dmso_features.cpu().numpy())
        self.mean_ = torch.tensor(pca.mean_, dtype=torch.float32)
        self.components_ = torch.tensor(pca.components_, dtype=torch.float32)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        Applies TVN to a set of input features using the fitted PCA model, returning the 
        whitened and decorrelated features.
        """
        x_centered = features - self.mean_.to(features.device)
        x_transformed = torch.matmul(x_centered, self.components_.T.to(features.device))
        return x_transformed

# not needed when using tvn in the extractor
def correct_tvn(features_DMSO, features_all):
    # Strip metadata columns if included
    labels = features_all[:, -6:]  # last 6 are metadata
    features_all_core = features_all[:, :-6]

    # Fit PCA on full DMSO features (no labels assumed)
    p = PCA(n_components=384, whiten=True)
    p.fit(features_DMSO)

    features_all_tvn = p.transform(features_all_core)
    return np.concatenate([features_all_tvn, labels], axis=1)