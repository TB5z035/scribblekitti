import torch
import torch.nn as nn
import torch.nn.functional as F


class TwinsLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def _assert_feat(self, feature_a, feature_b):
        batch_size_a, feature_size_a = feature_a.shape
        batch_size_b, feature_size_b = feature_b.shape
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
        feature_a = self.bn(feature_a)
        feature_b = self.bn(feature_b)

        cross_correlation = feature_a.T @ feature_b # [d, d]
        torch.distributed.all_reduce(cross_correlation) 
        cross_correlation.div_(torch.distributed.get_world_size())
        down = (feature_a.pow(2).sum(dim=0, keepdim=True).sqrt().T) @ (feature_b.pow(2).sum(dim=0, keepdim=True).sqrt())
        cross_correlation.div_(down)

        on_diag = torch.diagonal(cross_correlation).add_(-1).pow_(2).sum().div(feature_size)
        off_diag = off_diagonal(cross_correlation).pow_(2).sum().div((feature_size - 1) ** 2)

        loss = on_diag + self.coef * off_diag

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

        cov_a = (feature_a.T @ feature_a) / (batch_size - 1)
        down_a = (feature_a.pow(2).sum(dim=0, keepdim=True).sqrt().T) @ (feature_a.pow(2).sum(dim=0, keepdim=True).sqrt())
        cov_a.div_(down_a)
        cov_b = (feature_b.T @ feature_b) / (batch_size - 1)
        down_b = (feature_b.pow(2).sum(dim=0, keepdim=True).sqrt().T) @ (feature_b.pow(2).sum(dim=0, keepdim=True).sqrt())
        cov_b.div_(down_b)
        cov_loss = off_diagonal(cov_a).pow_(2).sum().div(
            (feature_size - 1) **2
        ) + off_diagonal(cov_b).pow_(2).sum().div((feature_size - 1) **2)

        print(repr_loss, std_loss, cov_loss)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss

class MECTwinsLoss(TwinsLoss):

    def __init__(self, num_features=128, n=4, **_) -> None:
        super().__init__()
        self.n = n
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)

    def forward(self, feature_a, feature_b):
        batch_size, feature_size = self._assert_feat(feature_a, feature_b)
        eps_d = 1e10 / feature_size
        self.lamb = 1 / (batch_size * eps_d)
        self.mu = (batch_size + feature_size) / 2
        c = self.bn(feature_a).T @ self.bn(feature_b) * self.lamb
        
        power_matrix = c
        sum_matrix = torch.zeros_like(power_matrix)

        for k in range(1, self.n + 1):
            if k > 1:
                power_matrix = torch.matmul(power_matrix, c)
            if (k + 1) % 2 == 0:
                sum_matrix += power_matrix / k
            else: 
                sum_matrix -= power_matrix / k
        
        loss = - torch.trace(sum_matrix) * self.mu
        
        return loss

def chunk_avg(x, n_chunks=2, normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)
  
class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        p, m = W.shape
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)

class Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)
        
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
            
        z_sim = z_sim/num_patch
        z_sim_out = z_sim.clone().detach()
                
        return -z_sim, z_sim_out

class EMPLoss(TwinsLoss):
    def __init__(self, num_features=128, num_patches=20, tcr=1, **_) -> None:
        super().__init__()
        self.TCR = TotalCodingRate()
        self.contractive_loss = Similarity_Loss()
        self.num_features = num_features
        self.num_patches = num_patches
        self.tcr = tcr
    
    def forward(self, features):
        batch_size, feature_size = features[0].shape
        feature_list = features.chunk(num_patches, dim=0)
        
        loss_TCR = 0
        for i in range(num_patches):
            loss_TCR += TCR(feature_list[i])
        loss_TCR = loss_TCR/num_patches
        
        z_list = z_proj.chunk(num_patches, dim=0)
        z_avg = chunk_avg(z_proj, num_patches)
        loss_contract, _ = self.contractive_loss(z_list, z_avg)
        
        loss = self.patch_sim * loss_contract + self.tcr * loss_TCR
        return loss