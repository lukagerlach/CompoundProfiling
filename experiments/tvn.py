import torch
from sklearn.decomposition import PCA

class TypicalVariationNormalizer:
    def __init__(self, eps=1e-8):
        self.pca = None
        self.mean_ = None
        self.components_ = None
        self.eigvals_ = None
        self.eps = eps

    def fit(self, dmso_features: torch.Tensor):
        """
        Fit PCA to the DMSO (negative control) embeddings.
        """
        n_samples, n_features = dmso_features.shape
        n_components = min(n_samples, n_features)
        self.pca = PCA(n_components=n_components, whiten=False)
        self.pca.fit(dmso_features.numpy())

        self.mean_ = torch.tensor(self.pca.mean_, dtype=torch.float32)
        self.components_ = torch.tensor(self.pca.components_, dtype=torch.float32)
        self.eigvals_ = torch.tensor(self.pca.explained_variance_, dtype=torch.float32).sqrt()
        self.eigvals_ = torch.clamp(self.eigvals_, min=self.eps)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply the TVN transform to the given features.
        """
        x_centered = features - self.mean_
        x_pca = torch.matmul(x_centered, self.components_.T)
        x_normalized = x_pca / self.eigvals_
        return x_normalized

    def fit_transform(self, dmso_features: torch.Tensor) -> torch.Tensor:
        self.fit(dmso_features)
        return self.transform(dmso_features)
