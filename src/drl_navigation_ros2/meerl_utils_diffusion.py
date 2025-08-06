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

class Model(nn.Module):
    def __init__(self, condition_dim, out_dim, hidden_size=256, time_dim=32):
        super(Model, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, time_dim),
        )

        input_dim = condition_dim + time_dim + out_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, out_dim)
        )
        self.apply(init_weights)

    def forward(self, x, time, state):
        t = self.time_mlp(time)
        out = torch.cat([x, t, state], dim=-1)
        out = self.layer(out)
        return out

class Model_LayerNorm(nn.Module):
    def __init__(self, condition_dim, out_dim, hidden_size=256, time_dim=32):
        super(Model_LayerNorm, self).__init__()

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

class MEECalculator(nn.Module):
    """
    最小误差熵(MEE)计算器，使用PyTorch实现并行计算
    """
    def __init__(self, bandwidth=0.1, device=None, epsilon=1e-10):
        """
        初始化MEE计算器
        
        参数:
            bandwidth: 高斯核函数的带宽参数
            device: 计算设备 (None表示使用当前设备)
            epsilon: 数值稳定性的小常数
        """
        super(MEECalculator, self).__init__()
        
        self.bandwidth = bandwidth
        self.epsilon = epsilon
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # 将参数移动到指定设备
        self.to(self.device)
        
    def forward(self, actions_list):

        # batch_size, num_nodes, action_dim = actions_list.shape
        
        # 使用广播计算所有批次的距离矩阵
        x1 = actions_list.unsqueeze(2)  # [batch_size, num_nodes, 1, action_dim]
        x2 = actions_list.unsqueeze(1)  # [batch_size, 1, num_nodes, action_dim]
        
        # 计算所有批次的平方距离矩阵
        dist_sq = torch.sum((x1 - x2)**2, dim=3)  # [batch_size, num_nodes, num_nodes]
        
        # 应用高斯核，使用固定带宽
        kernel_matrices = torch.exp(-dist_sq / (2 * self.bandwidth**2))  # [batch_size, num_nodes, num_nodes]
        
        # 添加小常数以保证数值稳定性
        kernel_matrices = kernel_matrices + self.epsilon
        
        # 获取批次大小和节点数量
        batch_size, num_nodes, _ = kernel_matrices.shape
        
        # 创建掩码排除对角线元素(自身与自身的比较)
        mask = 1.0 - torch.eye(num_nodes, device=kernel_matrices.device).unsqueeze(0)
        masked_matrices = kernel_matrices * mask
        
        # 计算每个批次的信息势(Information Potential)，使用适当的归一化因子
        ips = torch.sum(masked_matrices, dim=[1, 2]) / (num_nodes * (num_nodes - 1))  # [batch_size]
        
        # 计算每个批次的熵
        # mee_per_batch = -torch.log(ips)  # [batch_size]
        mee_per_batch = ips
        # 计算平均MEE
        total_mee = torch.mean(ips)
        
        return mee_per_batch, total_mee
    
    def compute_batch_mee(self, actions_list):
        """
        计算批次的MEE，只返回总MEE值
        
        参数:
            actions_list: 形状为 [batch_size, num_nodes, action_dim] 的张量
            
        返回:
            total_mee: 所有批次的平均MEE
        """
        _, total_mee = self.forward(actions_list)
        return total_mee

class MEECalculator_1D(nn.Module):
    """
    最小误差熵(MEE)计算器，使用PyTorch实现并行计算
    针对1维动作优化内存使用，避免原地操作导致的梯度问题
    """
    def __init__(self, bandwidth=0.1, device=None, epsilon=1e-10):
        """
        初始化MEE计算器
        
        参数:
            bandwidth: 高斯核函数的带宽参数
            device: 计算设备 (None表示使用当前设备)
            epsilon: 数值稳定性的小常数
        """
        super(MEECalculator_1D, self).__init__()
        
        self.bandwidth = bandwidth
        self.epsilon = epsilon
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # 将参数移动到指定设备
        self.to(self.device)
        
    def forward(self, actions_list):
        """
        计算1维动作样本的最小误差熵，避免原地操作
        
        参数:
            actions_list: 形状为 [batch_size, num_nodes, 1] 的张量
                          或 [batch_size, num_nodes] 的张量
            
        返回:
            mee_per_batch: 每个批次的MEE值
            total_mee: 所有批次的平均MEE
        """
        # 确保输入是torch张量并且在正确的设备上
        if not isinstance(actions_list, torch.Tensor):
            actions_list = torch.tensor(actions_list, dtype=torch.float32, device=self.device)
        elif actions_list.device != self.device:
            actions_list = actions_list.to(self.device)
        
        # 处理输入维度
        if actions_list.dim() == 2:
            # 如果输入是[batch_size, num_nodes]，添加一个维度
            actions_list = actions_list.unsqueeze(-1)  # [batch_size, num_nodes, 1]
        
        batch_size, num_nodes, action_dim = actions_list.shape
        assert action_dim == 1, "This implementation is optimized for 1D actions only"
        
        # 移除最后一个维度，简化计算
        actions_list = actions_list.squeeze(-1)  # [batch_size, num_nodes]
        
        # 使用广播计算所有批次的距离矩阵
        x1 = actions_list.unsqueeze(2)  # [batch_size, num_nodes, 1]
        x2 = actions_list.unsqueeze(1)  # [batch_size, 1, num_nodes]
        
        # 计算平方距离矩阵 - 避免原地操作
        diff = x1 - x2  # [batch_size, num_nodes, num_nodes]
        dist_sq = diff * diff  # [batch_size, num_nodes, num_nodes]
        
        # 应用高斯核 - 避免原地操作
        kernel_arg = -dist_sq / (2 * self.bandwidth**2)
        kernel_matrices = torch.exp(kernel_arg)
        
        # 添加小常数以保证数值稳定性 - 避免原地操作
        kernel_matrices_stable = kernel_matrices + self.epsilon
        
        # 计算信息势
        ips = torch.mean(kernel_matrices_stable, dim=[1, 2])  # [batch_size]
        
        # 计算熵
        mee_per_batch = -torch.log(ips)
        
        # 计算平均MEE
        total_mee = torch.mean(mee_per_batch)
        
        return mee_per_batch, total_mee
    
    def compute_batch_mee(self, actions_list):
        """
        计算批次的MEE，只返回总MEE值
        
        参数:
            actions_list: 形状为 [batch_size, num_nodes, 1] 的张量
                          或 [batch_size, num_nodes] 的张量
            
        返回:
            total_mee: 所有批次的平均MEE
        """
        _, total_mee = self.forward(actions_list)
        return total_mee


class MEECalculator_Huber(nn.Module):
    """
    使用掩码的最小误差熵(MEE)计算器，完全向量化实现
    """

    def __init__(self, bandwidth=0.1, device=None, epsilon=1e-10):
        super(MEECalculator_Huber, self).__init__()

        self.bandwidth = bandwidth
        self.epsilon = epsilon

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.to(self.device)

    def forward(self, q_values, actions, delta=1.0):
        batch_size, node_count, _ = q_values.shape
        # 去掉最后一个维度，简化计算
        q_values = q_values.squeeze(-1)  # [batch, node]

        # 计算每个批次的中位数 - 在 node 维度上
        q_median = q_values.median(dim=1, keepdim=True).values  # [batch, 1]

        # 计算每个 Q 值与其批次中位数的差值
        residuals = torch.abs(q_values - q_median)  # [batch, node]

        # 创建有效性掩码
        valid_mask = residuals <= delta  # [batch, node]
        
        # 计算每个批次中有效节点的数量
        valid_nodes_per_batch = valid_mask.sum(dim=1)  # [batch]
        
        # 计算平均有效节点数量
        avg_valid_nodes = valid_nodes_per_batch.float().mean()  # 标量

        # 创建一个填充了无穷大的距离矩阵
        inf_tensor = torch.full((batch_size, node_count, node_count), float('inf'), device=self.device)

        # 计算动作之间的距离矩阵
        x1 = actions.unsqueeze(2)  # [batch, node, 1, act_dim]
        x2 = actions.unsqueeze(1)  # [batch, 1, node, act_dim]
        dist_sq = torch.sum((x1 - x2) ** 2, dim=3)  # [batch, node, node]

        # 创建掩码矩阵，只考虑有效节点之间的距离
        mask_matrix = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)  # [batch, node, node]

        # 排除自身与自身的比较
        self_mask = ~torch.eye(node_count, dtype=torch.bool, device=self.device).unsqueeze(0)  # [1, node, node]
        mask_matrix = mask_matrix & self_mask  # [batch, node, node]

        # 应用掩码，将无效距离设为无穷大
        masked_dist = torch.where(mask_matrix, dist_sq, inf_tensor)

        # 应用高斯核，使用固定带宽
        kernel_matrices = torch.exp(-masked_dist / (2 * self.bandwidth ** 2))  # [batch, node, node]

        # 将无穷大距离处的核值设为0
        kernel_matrices = torch.where(masked_dist < float('inf'), kernel_matrices, torch.zeros_like(kernel_matrices))

        # 计算每个批次有效对的数量
        valid_pairs = mask_matrix.sum(dim=[1, 2])  # [batch]

        # 计算每个批次的信息势
        batch_ips = torch.sum(kernel_matrices, dim=[1, 2])  # [batch]

        # 标准化信息势（确保除数不为0）
        valid_pairs = torch.clamp(valid_pairs, min=1.0)  # 避免除零
        normalized_ips = batch_ips / valid_pairs  # [batch]

        # 计算平均MEE
        # 只考虑有效对数大于0的批次
        valid_batch_mask = valid_pairs > 0
        if valid_batch_mask.sum() > 0:
            total_mee = torch.mean(normalized_ips[valid_batch_mask])
        else:
            total_mee = torch.tensor(0.0, device=self.device)

        # return actions, valid_mask, normalized_ips, total_mee

        return normalized_ips, total_mee, avg_valid_nodes


if __name__ == "__main__":
    # 使用示例
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mee_calculator = MEECalculator_Huber(bandwidth=2, device=device)

    # 批次大小和节点数
    batch_size = 2
    node_count = 10
    act_dim = 3

    # 创建示例 Q 值 [batch, node, 1]
    # 批次1：正常分布加一些离群值
    q_batch1 = torch.tensor([2.0, 2.1, 1.9, 2.2, 2.0, 10.0, 1.8, 2.3, 1.7, 2.1],
                            dtype=torch.float32).view(1, 10, 1)

    # 批次2：正常分布加一些离群值
    q_batch2 = torch.tensor([5.0, 5.2, 4.8, 5.1, 15.0, 4.9, 5.3, 4.7, 5.2, 5.0],
                            dtype=torch.float32).view(1, 10, 1)

    # 合并批次
    q_values = torch.cat([q_batch1, q_batch2], dim=0).to(device)  # Shape: [2, 10, 1]

    # 创建示例动作 [batch*node, act_dim]
    actions = torch.randn(batch_size, node_count, act_dim).to(device)

    # 调用 MEECalculator
    filtered_actions, valid_mask, normalized_ips, total_mee, avg_valid_nodes = mee_calculator(q_values, actions, delta=1.0)

    print(f"批次1有效节点掩码: {valid_mask[0]}")
    print(f"批次2有效节点掩码: {valid_mask[1]}")
    print(f"批次1有效节点数: {valid_mask[0].sum().item()}")
    print(f"批次2有效节点数: {valid_mask[1].sum().item()}")
    print(f"批次1信息势: {normalized_ips[0].item():.6f}")
    print(f"批次2信息势: {normalized_ips[1].item():.6f}")
    print(f"平均MEE: {total_mee.item():.6f}")
    print(f"平均有效节点数量: {avg_valid_nodes.item():.6f}")

    # 验证掩码效果
    for b in range(batch_size):
        batch_q = q_values[b].squeeze()
        median = batch_q.median()
        residuals = torch.abs(batch_q - median)
        print(f"\n批次{b + 1}的Q值: {batch_q}")
        print(f"批次{b + 1}的中位数: {median.item():.4f}")
        print(f"批次{b + 1}的残差: {residuals}")
        print(f"批次{b + 1}有效样本: {batch_q[valid_mask[b]]}")

# 或者只获取总MEE:
# total_mee = mee_calculator.compute_batch_mee(actions_list)
# print(f"平均MEE: {total_mee.item():.6f}")

# 反向传播示例:
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer.zero_grad()
# total_mee = mee_calculator.compute_batch_mee(actions_list)
# total_mee.backward()  # 最小化MEE
# optimizer.step()
