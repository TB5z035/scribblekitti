import torch
import torch.nn as nn
import torch.nn.functional as F


class TwinsLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def _assert_feat(self, feature_a, feature_b):
        feature_size_a, batch_size_a = feature_a.shape
        feature_size_b, batch_size_b = feature_b.shape
        assert batch_size_a == batch_size_b, f"Batch size {batch_size_a} is not equal to {batch_size_b}"
        assert feature_size_a == feature_size_b, f"Feature size {feature_size_a} is not equal to {feature_size_b}"
        # feature_size = min(feature_size_a, feature_size_b)
        return batch_size_a, feature_size_a

    def forward(self, feature_a, feature_b):
        raise NotImplementedError


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(TwinsLoss):

    def __init__(self, num_features=128, coef=0.3, **_):
        super().__init__()
        self.coef = coef
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)

    def forward(self, feature_a, feature_b):
        # feature [B, d]
        # feature_a / feature_b -> (batch_size, feature_size)
        batch_size, feature_size = self._assert_feat(feature_a, feature_b)

        cross_correlation = self.bn(feature_a).T @ self.bn(feature_b) # [d, d]
        cross_correlation.div_(batch_size)
        torch.distributed.all_reduce(cross_correlation)

        on_diag = torch.diagonal(cross_correlation).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(cross_correlation).pow_(2).sum()
        loss = on_diag + self.coef * off_diag

        return loss


class MECTwinsLoss(TwinsLoss):

    def __init__(self, num_features=128, lamb=0.1, mu=0.1, n=4, **_) -> None:
        super().__init__()
        self.lamb = lamb
        self.mu = mu
        self.n = n
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)

    def forward(self, feature_a, feature_b):
        batch_size, feature_size = self._assert_feat(feature_a, feature_b)
        self.lamb = 1 / 0.06 / batch_size
        self.mu = (batch_size + feature_size) / 2
        cross_correlation = self.lamb * torch.mm(self.bn(feature_a).t(), self.bn(feature_b)) / batch_size

        sum_p = torch.zeros_like(cross_correlation)
        power = torch.clone(cross_correlation)
        for i in range(1, self.n + 1):
            if i > 1:
                power = torch.mm(power, cross_correlation)
            if i % 2 == 1:
                sum_p += power / i
            else:
                sum_p -= power / i
        loss = - self.mu * torch.trace(sum_p)
        
        return loss

class VICReg(TwinsLoss):
    def __init__(self, num_features=128, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, **_) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.num_features = num_features
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, feature_a, feature_b):
        batch_size, feature_size = self._assert_feat(feature_a, feature_b)
        
        # invariance: same point, two transforms
        
        repr_loss = F.mse_loss(feature_a, feature_b)

        feature_a = feature_a - feature_a.mean(dim=0)
        feature_b = feature_b - feature_b.mean(dim=0)
        
        # variance: within batch

        std_a = torch.sqrt(feature_a.var(dim=0) + 0.0001)
        std_b = torch.sqrt(feature_b.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_a)) / 2 + torch.mean(F.relu(1 - std_b)) / 2
        
        # covariance: within batch

        cov_a = (feature_a.T @ feature_a) / (feature_size - 1)
        cov_b = (feature_b.T @ feature_b) / (feature_size - 1)
        cov_loss = off_diagonal(cov_a).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_b).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss