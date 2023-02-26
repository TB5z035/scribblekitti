import torch
import torch.nn as nn


class TwinsLoss:

    def __call__(self, feature_a, feature_b):
        return self.forward(feature_a, feature_b)

    def _assert_feat(self, feature_a, feature_b):
        batch_size_a, feature_size_a = feature_a.shape
        batch_size_b, feature_size_b = feature_b.shape
        assert batch_size_a == batch_size_b, f"Batch size {batch_size_a} is not equal to {batch_size_b}"
        assert feature_size_a == feature_size_b, f"Feature size {feature_size_a} is not equal to {feature_size_b}"
        return batch_size_a, feature_size_a

    def forward(self, feature_a, feature_b):
        raise NotImplementedError


class BarlowTwinsLoss(TwinsLoss):

    def __init__(self, coef=0.3):
        self.coef = coef

    def forward(self, feature_a, feature_b):
        # feature_a / feature_b -> (batch_size, feature_size)
        batch_size, feature_size = self._assert_feat(feature_a, feature_b)

        feature_a = (feature_a - feature_a.mean(dim=0)) / (feature_a.std(dim=0) + 1e-6)
        feature_b = (feature_b - feature_b.mean(dim=0)) / (feature_b.std(dim=0) + 1e-6)
        cross_correlation = torch.mm(feature_a.t(), feature_b) / batch_size

        remain = (cross_correlation - torch.eye(feature_size, device=cross_correlation.device)).pow(2)
        remain[(1 - torch.eye(feature_size, device=remain.device)).to(torch.bool)] *= self.coef
        loss = remain.sum()

        return loss


class MECTwinsLoss(TwinsLoss):

    def __init__(self, lamb=0.1, mu=0.1, n=10) -> None:
        self.lamb = lamb
        self.mu = mu
        self.n = n

    def forward(self, feature_a, feature_b):
        batch_size, feature_size = self._assert_feat(feature_a, feature_b)
        self.lamb = 1 / 0.06 / batch_size   
        self.mu = (batch_size + feature_size) / 2

        feature_a = (feature_a - feature_a.mean(dim=0)) / feature_a.std(dim=0)  
        feature_b = (feature_b - feature_b.mean(dim=0)) / feature_a.std(dim=0)
        cross_correlation = torch.mm(feature_a.t(), feature_b) / batch_size

        sum_p = torch.zeros_like(cross_correlation)
        power = torch.clone(cross_correlation)
        for i in range(self.n + 1):
            if i > 1:
                power = torch.mm(power, cross_correlation)
            if (i + 1) % 2 == 0:
                sum_p += power / i
            else:
                sum_p -= power / i
        loss = -self.mu * torch.trace(sum_p)
        return loss

