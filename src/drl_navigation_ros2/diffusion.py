import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils_diffusion import get_beta_schedule, Model, extract


class Diffusion(nn.Module):

    def __init__(self, condition_dim, out_dim, noise_ratio=1.0, beta_schedule='vp', n_timesteps=30,
                 predict_epsilon=False, device='cuda'):
        super().__init__()
        self.condition_dim = condition_dim
        self.out_dim = out_dim
        self.model = Model(condition_dim, out_dim).to(device)
        self.device = device  # 添加 device 参数

        betas = get_beta_schedule(beta_schedule, n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.n_timesteps = int(n_timesteps)
        self.noise_ratio = noise_ratio
        self.predict_epsilon = predict_epsilon
        self.register_buffer('alphas', alphas.to(device))
        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.to(device))
        self.register_buffer('sqrt_alphas_cumprod_prev', torch.sqrt(alphas_cumprod_prev).to(device))
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).to(device))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod).to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).to(device))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance.to(device))

        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)).to(device))
        self.register_buffer('posterior_mean_coef1',
                             (betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(device))
        self.register_buffer('posterior_mean_coef2',
                             ((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(device))

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape).to(self.device) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape).to(self.device) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape).to(self.device) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape).to(self.device) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape).to(self.device)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape).to(self.device)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))
        # if self.clip_denoised:
        #     x_recon.clamp_(-1., 1.)
        # else:
        #     assert RuntimeError()
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)

        noise = torch.randn_like(x, device=self.device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).to(self.device)

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise * self.noise_ratio


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape).to(self.device) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape).to(self.device) * noise
        )

        return sample

    def ddim_sample_loop(self, state, shape, stride, K_repeat=1):
        device = self.device

        raw_batch_size = shape[0]
        shape = list(shape)  # 将 torch.Size 转换为列表
        shape[0] = shape[0] * K_repeat  # 修改 shape[0]

        shape = torch.Size(shape)  # 将列表转换回 torch.Size
        batch_size = shape[0]
        x_i = torch.randn(shape, device=device)

        state = state.repeat(K_repeat, 1)
        # for i in range(self.n_timesteps, 0, -stride):
        for i in reversed(range(0, self.n_timesteps, stride)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            noise_x = self.model(x_i, timesteps, state)
            alpha_bar = extract(self.alphas_cumprod, timesteps, x_i.shape).to(self.device)
            if (i - stride) > 0:
                alpha_bar_prev = extract(self.alphas_cumprod, timesteps - stride, x_i.shape).to(self.device)
            else:
                alpha_bar_prev = torch.full(x_i.shape, 1.0, device=self.device)
            if self.predict_epsilon:
                x_i = alpha_bar_prev * (x_i - (1 - alpha_bar).sqrt() * noise_x) / alpha_bar.sqrt() + (1 - alpha_bar_prev).sqrt() * noise_x
            else:
                c0 = ((1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt()
                x_i = alpha_bar_prev.sqrt() * noise_x + c0 * (x_i - alpha_bar.sqrt() * noise_x)
        x_i = x_i.view(K_repeat, raw_batch_size, -1).transpose(0, 1)
        return x_i


    def p_sample_loop(self, state, shape, K_repeat=1):
        device = self.device
        raw_batch_size = shape[0]

        shape = list(shape)  # 将 torch.Size 转换为列表
        shape[0] = shape[0] * K_repeat  # 修改 shape[0]
        shape = torch.Size(shape)  # 将列表转换回 torch.Size

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        state = state.repeat(K_repeat, 1)
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
        x = x.view(K_repeat, raw_batch_size, -1).transpose(0,1)

        return x
    def sample(self, state, shape, K_repeat=1, type="ddpm", stride=10):
        if type == "ddpm":
            return self.p_sample_loop(state, shape, K_repeat)
        elif type == "ddim":
            return self.ddim_sample_loop(state, shape, stride=stride, K_repeat=K_repeat)


    def get_loss(self, x_start, state):
        batch_size = len(x_start)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn_like(x_start, device=self.device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = F.mse_loss(x_recon, noise, reduction='mean')
        else:
            loss = F.mse_loss(x_recon, x_start, reduction='mean')
        return loss

    def get_w_loss(self, x_start, state, w):
        batch_size = len(x_start)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn_like(x_start, device=self.device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = F.mse_loss(x_recon, noise, reduction='none')
        else:
            loss = F.mse_loss(x_recon, x_start, reduction='none')

        w = torch.exp(w)  # 确保权重为正数
        weighted_loss = (loss * w).mean()  # 加权后求平均损失
        return weighted_loss






