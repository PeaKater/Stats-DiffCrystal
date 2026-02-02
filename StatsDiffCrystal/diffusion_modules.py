import math
import torch
import numpy as np
import torch.nn as nn
import cupy as cp

from torch.nn import functional as F
from scipy import interpolate
from scipy.special import ive,i0e
from gnn.embeddings import MAX_ATOMIC_NUM
from tqdm import tqdm


class SSDDPM(nn.Module):
    def __init__(self, timesteps, type_sigma_begin, type_sigma_end,
                 same_time_step=False):
        super().__init__()
        self.num_steps = timesteps
        self.same_time_step = same_time_step
        
        #noising for Type
        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(type_sigma_begin),
            np.log(type_sigma_end),
            timesteps)),dtype=torch.float32)
        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)
        # cumsum
        def reverse_cumsum(series):
            return torch.flip(torch.cumsum(torch.flip(series, dims=[0]), dim=0),dims=[0])
        self.reverse_cumsum = reverse_cumsum
        
        # noising for frac_coords
        kappa_t = nongaussian_scheduler(self.num_steps)
        kappa_t_1 = torch.hstack((torch.tensor([1.1*kappa_t[0]]), kappa_t[:-1]))
        self.kappa_t = nn.Parameter(kappa_t.reshape(-1,1), requires_grad=False)
        self.kappa_t_1 = nn.Parameter(kappa_t_1.reshape(-1,1), requires_grad=False)
        
        # KL divergence coefficient
        Ive_1 = ive(1+torch.zeros_like(kappa_t),kappa_t)
        Ive_0 = i0e(kappa_t)
        kl_coef = torch.log10(kappa_t * Ive_1/Ive_0)
        min, max = -1,1
        kl_coef = 10**(min+((kl_coef-torch.min(kl_coef))*(max-min)/(torch.max(kl_coef)-torch.min(kl_coef))))
        self.kl_coef = nn.Parameter(kl_coef.reshape(-1,1), requires_grad=False)

        #static variables
        self.at = self.kappa_t
        self.T = lambda xt: (torch.cos(xt), torch.sin(xt))
        self.to_domain = lambda xt: 2*torch.pi*(xt-0.5)     #[0,1]--[-pi,pi]
        self.from_domain = lambda xt: xt/(2*torch.pi)+0.5   #[-pi,pi]--[0,1]

    def get_Gt_mean_std(self, sample=False):
        if not sample:
            self.gt_norm = self.reverse_cumsum(self.kappa_t)
        else:    
            self.gt_norm = self.reverse_cumsum(self.kappa_t_1)
        
    # Forward process
    def perturb_sample(self, r, Z, composition_probs, num_atoms, t=None):
        lowest_t = 1
        if t is None:
            t = torch.randint(
                lowest_t, self.num_steps, size=(num_atoms.size(0),), device=r.device)
            if self.same_time_step:
                s = t
            else:
                s = torch.randint(
                    lowest_t, self.num_steps, size=(num_atoms.size(0),), device=r.device)
        else:
            assert isinstance(t, int)
            t = torch.tensor([t]*num_atoms.size(0), device=r.device).long()
            s = torch.randint(
                    lowest_t, self.num_steps, size=(num_atoms.size(0),), device=r.device)
        
        t_int = t.repeat_interleave(num_atoms, dim=0)
        s_int = s.repeat_interleave(num_atoms, dim=0)
        
        sigma_type_s = self.type_sigmas[s_int][:,None]
        A_s = F.one_hot(Z-1, num_classes=MAX_ATOMIC_NUM)+composition_probs*sigma_type_s
        A_s = A_s / (A_s.sum(-1, keepdim=True) + 1e-8)

        r_theta = self.sample_Gt(r, t_int)
        return (r_theta, A_s), (t, s[:,None], t_int)
    
    def sample_Gt(self, r, t):
        N = self.num_steps
        bs = r.shape[0]
        r = self.to_domain(r)
        
        expand = lambda series: series.flatten().repeat(bs).reshape(bs,N).T.flatten()
        all_time = expand(torch.arange(N,device=r.device))
        xs = self.forward_step_sample(r.repeat(N,1), all_time)
        
        cosGt,sinGt = self.tail_statistic_term(xs, all_time)
        cosGt = cosGt.reshape(N, bs, 3)
        sinGt = sinGt.reshape(N, bs, 3)

        Gt_idx = (all_time.reshape(N, bs)>=t.squeeze()).reshape(N,bs,1)
        cosGt,sinGt = torch.sum(cosGt*Gt_idx,0),torch.sum(sinGt*Gt_idx,0)
        theta = torch.atan2(sinGt, cosGt)
        return self.from_domain(theta)
    
    def forward_step_sample(self, x0, t):
        mu, kappa = self.forward_step_distribution(x0,t)
        return self.from_domain(self.sample(mu,kappa))
    
    def forward_step_distribution(self, x0, t):
        return x0, self.kappa_t[t]
    
    def sample(self, mu, kappa):
        return VonMises(mu, kappa).sample()
    
    def tail_statistic_term(self, xt, t):
        cos_xt, sin_xt = self.T(self.to_domain(xt))
        b = cos_xt.size(0)
        return self.at[t].view(b,1) * cos_xt, self.at[t].view(b,1) * sin_xt
    
    def tail_normalization(self,cGt,sGt,t):
        return (cGt/self.gt_norm[t], sGt/self.gt_norm[t])
    ##LOSS function
    def kl(self, x0, x0_pred, t):
        x0, x0p = self.to_domain(x0), self.to_domain(x0_pred)
        return torch.sum(self.kl_coef[t]*(1-torch.cos(x0-x0p)), dim=-1)
   
    ##sample function
    def result_distribution(self, num_atoms):
        bs, dev = num_atoms.sum(), num_atoms.device
        mu = (torch.rand((bs,3), dtype=torch.float32, device=dev)*2-1)*torch.pi
        t = torch.tensor([self.num_steps-1], device=dev).repeat(bs)
        return (mu,self.kappa_t[t])
    
    def result_distribution_sample(self, num_atoms):
        return self.from_domain(self.sample(*self.result_distribution(num_atoms)))
    
    def update_Gt(self, xt, G_c_t, G_s_t, t, num_atoms=None):
        cosGt, sinGt = self.tail_statistic_term(xt, t)
        cosGt = cosGt.reshape(xt.shape[0], 3)
        sinGt = sinGt.reshape(xt.shape[0], 3)
        cosGt, sinGt = cosGt+G_c_t, sinGt+G_s_t
        theta = torch.atan2(sinGt, cosGt)
        return cosGt, sinGt, self.from_domain(theta)
    
    def reverse_step_distribution(self, x0, t):
        return (x0, self.kappa_t_1[t]/0.95)
    
    def reverse_step_sample(self, x0, t, num_atoms=None):
        x0 = self.to_domain(x0)
        mu, kappa = self.reverse_step_distribution(x0, t)
        return self.from_domain(self.sample(mu,kappa))
    def forward(self):
        return NotImplementedError


class VonMises:
    """
    DESCRIPTION:
        Class of von Mises distribution. Use CuPy for GPU-based sampling.
    """
    def __init__(self, mu, kappa):
        self.mu = mu
        self.kappa = kappa.view(mu.size(0),1).repeat(1,mu.size(1))
        self.device = mu.device

    def sample(self, size=None):
        if size is None:
            size = (self.mu.size(0),3)
        samples_cp = cp.random.vonmises(cp.asarray(self.mu),
                                        cp.asarray(self.kappa),
                                        size=size)
        samples = torch.tensor(samples_cp, device=self.device, dtype=torch.float32)
        return samples


def nongaussian_scheduler(num_steps):
    T = 100
    fit_params_y = [3,1.3,0.6,0.3,-0.3,-3]
    fit_params_x = np.array([0.01,0.14,0.35,0.6,0.8,1])
    start_kappa_pow = 3
    kappa_0, kappa_T = np.max(fit_params_y), np.min(fit_params_y)
    fit_y_scaled = (fit_params_y - kappa_T) * ((start_kappa_pow - kappa_T) / (kappa_0 - kappa_T)) + kappa_T
    spline = interpolate.CubicSpline(T * fit_params_x, fit_y_scaled)
    log10_theta = torch.tensor(spline(T * np.arange(num_steps) / num_steps), dtype=torch.float32)
    kappa_t = torch.pow(10, log10_theta)
    return kappa_t