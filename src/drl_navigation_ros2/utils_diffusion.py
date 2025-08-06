import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear:
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)

import torch
import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self, condition_dim, out_dim, hidden_size=256, time_dim=32):
#         super(Model, self).__init__()
#
#         self.time_mlp = nn.Sequential(
#             SinusoidalPosEmb(time_dim),
#             nn.Linear(time_dim, hidden_size),
#             nn.Mish(),
#             nn.Linear(hidden_size, time_dim),
#         )
#
#         input_dim = condition_dim + time_dim + out_dim
#         self.layer = nn.Sequential(
#             nn.Linear(input_dim, hidden_size),
#             nn.Mish(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.Mish(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.Mish(),
#             nn.Linear(hidden_size, out_dim)
#         )
#         self.apply(init_weights)
#
#     def forward(self, x, time, state):
#         t = self.time_mlp(time)
#         out = torch.cat([x, t, state], dim=-1)
#         out = self.layer(out)
#         return out

class Model(nn.Module):
    def __init__(self, condition_dim, out_dim, hidden_size=256, time_dim=32):
        super(Model, self).__init__()

        # time_mlp：对 time 输入做编码
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_size),
            nn.Mish(),
            nn.LayerNorm(hidden_size),    # 对隐藏层输出做 LayerNorm
            nn.Linear(hidden_size, time_dim),
        )

        # 主网络部分
        input_dim = condition_dim + time_dim + out_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),    # LayerNorm
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),    # LayerNorm
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),    # LayerNorm
            nn.Mish(),
            nn.Linear(hidden_size, out_dim)
        )

        # 自定义初始化
        self.apply(init_weights)

    def forward(self, x, time, state):
        # 用 time_mlp 对时间步骤做编码
        t = self.time_mlp(time)
        # 将 x, t, state 拼接
        out = torch.cat([x, t, state], dim=-1)
        # 通过主网络
        out = self.layer(out)
        return out

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)


def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def get_beta_schedule(schedule_name, n_timesteps):
    if schedule_name == 'linear':
        betas = linear_beta_schedule(n_timesteps)
    elif schedule_name == 'cosine':
        betas = cosine_beta_schedule(n_timesteps)
    elif schedule_name == 'vp':
        betas = vp_beta_schedule(n_timesteps)
    else:
        raise NotImplementedError(f"未知的 beta 计划：{schedule_name}")
    return betas

import torch


import torch
import math

import math
import torch


class DistributionMetrics:
    def __init__(
            self,
            batch_size,
            num_nodes,
            feature_dim,
            scale=0.5,
            bandwidth=0.1,
            uniform_range=(-1, 1),
            device="cuda",
    ):
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.bandwidth = bandwidth
        self.uniform_range = uniform_range
        self.device = device
        # volume = (uniform_range[1] - uniform_range[0]) ** feature_dim
        # self.uniform_pdf = 1.0 / volume  # q(x) = constant
        # # 预先缓存 log_q，避免每次重复调用 math.log()
        # self.log_uniform_pdf = math.log(self.uniform_pdf)

        # 归一化因子： (2π)^(d/2) * (bandwidth^d)
        self.normalization_factor = (2 * math.pi) ** (feature_dim / 2) * (
                bandwidth ** feature_dim
        )
        samples = np.random.uniform(
            low=uniform_range[0],
            high=uniform_range[1],
            size=(batch_size, num_nodes, feature_dim)
        )
        data_torch = torch.from_numpy(samples).float().view(batch_size, num_nodes, feature_dim)
        self.kde_pdf_values = self.kde_pdf(data_torch, data_torch)
        self.log_uniform_pdf = math.log(self.kde_pdf_values.mean(dim=[0,1]))

        samples_max = np.random.uniform(
            low=uniform_range[1],
            high=uniform_range[1],
            size=(batch_size, num_nodes, feature_dim)
        )
        data_torch_max = torch.from_numpy(samples_max).float().view(batch_size, num_nodes, feature_dim)
        self.kde_pdf_values_max = self.kde_pdf(data_torch_max, data_torch_max)
        self.log_uniform_pdf_max = math.log(self.kde_pdf_values_max.mean(dim=[0,1]))
        max_scale = self.log_uniform_pdf_max / self.log_uniform_pdf
        target_scale = (max_scale - 1) * scale + 1
        self.log_uniform_pdf = self.log_uniform_pdf * target_scale
        print("target")
    #
    def kde_pdf(self, batch_data, ref_data):
        """
        参数:
        - batch_data: [batch, N, feature_dim]
        - ref_data:   [batch, N, feature_dim] (或与 batch_data 形状相同/相似)

        返回值:
        - pdf_values: [batch, N]
        """
        # 使用 torch.cdist 计算 pairwise distance
        # cdist 会返回形状 [batch, N, N]，表示 batch 内部每个样本之间的距离
        distances = torch.cdist(batch_data, ref_data)  # [batch, N, N]
        distances_scaled = distances / self.bandwidth  # [batch, N, N]

        # (distances_scaled^2) -> [batch, N, N]
        distances_squared = distances_scaled.pow(2)

        # 高斯核
        gaussian_kernel = torch.exp(-0.5 * distances_squared)  # [batch, N, N]

        # 对第 -1 维(N)求和，再除以归一化因子和节点数
        pdf_values = gaussian_kernel.sum(dim=-1) / (self.num_nodes * self.normalization_factor)  # [batch, N]

        return pdf_values

    def entropy(self, kde_pdf_values):
        """
        估计熵:
        H(X) = - E_p[ log p(X) ] ≈ -1/N * Σ_i log p(x_i)
        最后对 batch 也取平均
        """
        log_p = torch.log(kde_pdf_values + 1e-12)  # [batch, N]
        # 对每个 batch 求 -平均值，然后再对所有 batch 求平均
        entropy_per_batch = -torch.mean(log_p, dim=1)
        return entropy_per_batch

    def kl_divergence(self, kde_pdf_values):
        """
        KL 散度:
        D_KL(p || q) = E_p[ log p(X) - log q(X) ]
                     = 1/N * Σ_i [log p(x_i) - log q(x_i)]
        最后对 batch 也取平均
        """
        log_p = torch.log(kde_pdf_values + 1e-12)  # [batch, N]
        # 利用预先缓存的 self.log_uniform_pdf
        kl_per_batch = torch.mean(log_p, dim=1)
        return torch.mean(kl_per_batch)


class DistributionMetrics_Beta:
    def __init__(
        self,
        batch_size,
        num_nodes,
        feature_dim,
        scale=0.5,
        bandwidth=0.1,
        uniform_range=(-1, 1),
        device="cuda",
    ):
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.bandwidth = bandwidth
        self.uniform_range = uniform_range
        self.device = device

        # ------------------------
        # 以下这个 normalization_factor 原本是高斯核用的，对于 Beta 核不再需要
        # 但为了保证兼容性，如果仍要用 self.kde_pdf(高斯核)，这里继续保留
        # (2π)^(d/2) * (bandwidth^d)
        self.normalization_factor = (2 * math.pi) ** (feature_dim / 2) * (
            bandwidth ** feature_dim
        )

        # ------------------------
        # 演示用：随机采样一些数据做测试
        samples = np.random.uniform(
            low=uniform_range[0],
            high=uniform_range[1],
            size=(batch_size, num_nodes, feature_dim)
        )
        data_torch = torch.from_numpy(samples).float().view(batch_size, num_nodes, feature_dim)

        # 这里先演示 Beta 核 (kde_pdf_beta)，并缓存其均值 -> 用来模拟“对数均值”
        beta_vals = self.kde_pdf_beta(data_torch, data_torch)  # [batch, N]
        self.log_uniform_pdf_init = torch.log(beta_vals.mean(dim=[0,1]))

        # 做一点类似原先对端点采样的逻辑（仅作演示）
        samples_max = np.random.uniform(
            low=1,
            high=1,
            size=(batch_size, num_nodes, feature_dim)
        )
        data_torch_max = torch.from_numpy(samples_max).float().view(batch_size, num_nodes, feature_dim)
        beta_vals_max = self.kde_pdf_beta(data_torch_max, data_torch_max)  # [batch, N]
        self.log_uniform_pdf_max = torch.log(beta_vals_max.mean(dim=[0,1]))

        max_scale = self.log_uniform_pdf_max / self.log_uniform_pdf_init
        target_scale = (max_scale - 1) * scale + 1
        self.log_uniform_pdf = self.log_uniform_pdf_init * target_scale

        print("Init done (Beta kernel).")

    def kde_pdf_beta(self, batch_data, ref_data, eps=1e-7):
        r"""
        使用 Beta 核进行 n 维 KDE，假设数据都在 [-1,1]^n。
        对于每个批次 b，每个目标点 i，需要对参考点 j 做如下运算：

            1) 将 batch_data[b,i,:] 从 [-1,1]^n 映射到 [0,1]^n  (记为 x_ij^*)
            2) 将 ref_data[b,j,:]   同样映射到 [0,1]^n  (记为 x_j^*)
            3) 在每一维 d=1..n 上构造 Beta( alpha_j[d], beta_j[d] )，其中
                   alpha_j[d] = x_j^*[d] / bandwidth + 1
                   beta_j[d]  = (1 - x_j^*[d]) / bandwidth + 1
            4) 计算 Beta PDF 在 x_ij^*[d] 处的值，然后对 d=1..n 做连乘 -> K_ij
            5) 对 j=1..N 做加和 / N -> 最终得到 p(x_i)。

        输入:
          - batch_data: [batch, N, feature_dim]
          - ref_data  : [batch, N, feature_dim]
        输出:
          - pdf_values:  [batch, N], 每个 batch 的每个样本点 i 对应的密度估计。

        注意:
          - 这是一个“乘积 Beta 核”，在每个维度上的 Beta 分布是独立相乘。
          - 与 Gaussian KDE 不同，这里不需要另行乘 (2π)^(d/2)*(h^d) 之类的归一化因子，
            Beta 分布本身在 [0,1] 上已经是归一化的。
        """
        # 转到同一个 device
        batch_data = batch_data.to(self.device)
        ref_data   = ref_data.to(self.device)

        B, N, D = batch_data.shape  # batch, num_nodes, feature_dim
        # 1) 将 [-1,1]^D -> [0,1]^D
        x_batch_01 = 0.5 * (batch_data + 1.0)
        x_ref_01   = 0.5 * (ref_data   + 1.0)




        # 2) 做广播运算: 我们想要最终得到 [B, N, N], 其中第 2 维是 "目标点 i", 第 3 维是 "参考点 j"
        #   x_batch_01[..., None, :] -> [B, N, 1, D]
        #   x_ref_01[...,  None, :]  -> [B, N, 1, D], 不过我们要让它变成 [B, 1, N, D]
        #   因此:
        x_batch_4d = x_batch_01.unsqueeze(2)  # [B, N, 1, D]
        x_ref_4d   = x_ref_01.unsqueeze(1)    # [B, 1, N, D]

        # 3) 计算 Beta 分布的 α, β
        #    alpha, beta 的 shape: [B, 1, N, D] (与 x_ref_4d 相同)
        alpha = x_ref_4d / self.bandwidth + 1.0
        beta_ = (1.0 - x_ref_4d) / self.bandwidth + 1.0

        # 4) 对于 x_batch_4d (目标点), 我们在每个维度上计算 Beta(α, β) 的对数 PDF 值，然后对 D 维求和，再 exp
        #    Beta(α, β) pdf = Gamma(α+β)/Gamma(α)/Gamma(β) * x^(α-1) * (1-x)^(β-1)
        #    在多维时: 取连乘 => 取对数后相加
        x_batch_4d_clamped = torch.clamp(x_batch_4d, eps, 1.0 - eps)

        log_beta_top = torch.lgamma(alpha + beta_)
        log_beta_bot = torch.lgamma(alpha) + torch.lgamma(beta_)

        # log_pdf_each_dim 的 shape: [B, N, N, D]
        #  (batch_broadcast, i, j, dim)
        log_pdf_each_dim = (
            log_beta_top - log_beta_bot
            + (alpha - 1.0) * torch.log(x_batch_4d_clamped)
            + (beta_ - 1.0) * torch.log(1.0 - x_batch_4d_clamped)
        )

        # sum over dim=D
        # shape -> [B, N, N]
        log_pdf_product = log_pdf_each_dim.sum(dim=-1)

        # exp => 得到乘积的 Beta PDF
        pdf_product = torch.exp(log_pdf_product)  # [B, N, N]

        # 5) 沿着第 2 维(参考点 j) 求和并做平均
        #    pdf_values[b, i] = 1/N * Σ_j pdf_product[b, i, j]
        pdf_values = pdf_product.sum(dim=2) / N  # [B, N]

        return pdf_values

    def kde_pdf(self, batch_data, ref_data):
        """
        (原) 高斯核版本的多维 KDE, 与之前相同。
        """
        distances = torch.cdist(batch_data, ref_data)  # [batch, N, N]
        distances_scaled = distances / self.bandwidth  # [batch, N, N]
        distances_squared = distances_scaled.pow(2)
        gaussian_kernel = torch.exp(-0.5 * distances_squared)  # [batch, N, N]

        pdf_values = gaussian_kernel.sum(dim=-1) / (self.num_nodes * self.normalization_factor)
        return pdf_values

    def entropy(self, kde_pdf_values):
        """
        估计熵:
          H(X) = - E_p[ log p(X) ]
               ≈ -1/N * Σ_i log p(x_i)
        最后对 batch 也取平均
        """
        log_p = torch.log(kde_pdf_values + 1e-12)  # [batch, N]
        entropy_per_batch = -torch.mean(log_p, dim=1)  # 每个batch
        return entropy_per_batch

    def kl_divergence(self, kde_pdf_values):
        """
        KL 散度:
        D_KL(p || q) = E_p[ log p(X) - log q(X) ]
                     = 1/N * Σ_i [log p(x_i) - log q(x_i)]
        最后对 batch 也取平均
        """
        log_p = torch.log(kde_pdf_values + 1e-12)  # [batch, N]
        # 这里的 self.log_uniform_pdf 若对应 Beta KDE, 需根据实际 "均匀参考分布" 再做定义
        kl_per_batch = torch.mean(log_p, dim=1)
        return torch.mean(kl_per_batch)

