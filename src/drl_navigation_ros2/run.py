import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboard.compat.tensorflow_stub.dtypes import float32

from ros_python import ROS_env
from TD3.TD3 import TD3
import torch.optim as optim
import numpy as np
import glob
import time
import math
from diffusion import Diffusion
from utils_diffusion import DistributionMetrics_Beta

class ActorDiffusion(nn.Module):
    def __init__(self, action_dim ,state_dim ,max_action,min_action):
        super().__init__()
        condition_dim = np.array(state_dim).prod()
        self.actor_dim = np.array(action_dim).prod()
        # self.actor = Diffusion(condition_dim, out_dim=self.actor_dim, predict_epsilon=True, n_timesteps=5).to(device)
        self.entropy_type = "KL"
        self.eval_sample = 1
        self.actor_noise = False
        self.eval_noise = False
        self.behavior_sample = 4
        self.num_nodes = 10
        self.actor = Diffusion(condition_dim, out_dim=self.actor_dim, predict_epsilon=False, n_timesteps=5,device='cpu').to('cpu')
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((max_action - min_action) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((max_action + min_action) / 2.0, dtype=torch.float32)
        )
        self.kde = DistributionMetrics_Beta(256, self.num_nodes , self.actor_dim, bandwidth=0.1, scale=0.01)


    def calculate_graph_smoothness_einsum(self, batch_data, q=None):
        pdf = self.kde.kde_pdf_beta(batch_data, batch_data)
        kl_gaussian = self.kde.kl_divergence(pdf)


        return kl_gaussian
    def calculate_actor_entropy(self, batch_data):
        pdf = self.kde.kde_pdf(batch_data, batch_data)
        return self.kde.entropy(pdf)


    def forward(self, state, q_func1=None, q_func2=None, stride=1):
        raw_batch_size = state.shape[0]
        action = self.get_K_actions(state, k_repeat=self.behavior_sample, stride=1)
        action = action.squeeze(0)
        if self.actor_noise:
            noise_std = self.action_scale * 0.1
            noise = torch.randn_like(action) * noise_std
            action = action + noise
        action = torch.clamp(action,
                             min=-1,  # 对应 env low
                             max=1)  # 对应 env high

        state = state.repeat(self.behavior_sample, 1)
        q1 = q_func1(state, action)
        q2 = q_func2(state, action)
        q = torch.min(q1, q2)
        action = action.view(self.behavior_sample, raw_batch_size, -1).transpose(0, 1)
        q = q.view(self.behavior_sample, raw_batch_size, -1).transpose(0, 1)
        action_idx = torch.argmax(q, dim=1, keepdim=True).repeat(1, 1, self.actor_dim)
        x = action.gather(dim=1, index=action_idx).view(raw_batch_size, -1)
        # x = x * self.action_scale + self.action_bias
        return x

    @torch.no_grad()
    def get_max_action(self, state, q_func1=None, q_func2=None, stride=1):
        raw_batch_size = state.shape[0]
        action = self.get_K_actions(state, k_repeat=self.eval_sample, stride=1)
        action = action.squeeze(0)
        if self.eval_noise:
            noise_std = self.action_scale * 0.1
            noise = torch.randn_like(action) * noise_std
            action = action + noise
        action = torch.clamp(action,
                             min=-1,  # 对应 env low
                             max=1)  # 对应 env high
        state = state.repeat(self.eval_sample, 1)
        q1 = q_func1(state, action)
        q2 = q_func2(state, action)
        q = torch.min(q1, q2)
        action = action.view(self.eval_sample, raw_batch_size, -1).transpose(0, 1)
        q = q.view(self.eval_sample, raw_batch_size, -1).transpose(0, 1)
        action_idx = torch.argmax(q, dim=1, keepdim=True).repeat(1, 1, self.actor_dim)
        x = action.gather(dim=1, index=action_idx).view(raw_batch_size, -1)
        # x = x * self.action_scale + self.action_bias
        return  x


    def get_K_actions(self, x, k_repeat=1, stride=1, is_graph=True):
        x = self.actor.sample(x, shape=[x.shape[0], self.actor_dim], K_repeat=k_repeat, type="ddpm", stride=stride).squeeze(1)
        # x = torch.tanh(x)
        if self.actor_noise:  # 添加噪声
            noise_std = self.action_scale * 0.1
            noise = torch.randn_like(x) * noise_std
            x = x + noise
        x = torch.clamp(x, min=-1.0, max=1.0)
        # actor_list =  x * self.action_scale + self.action_bias
        actor_list = x
        if k_repeat == 1:
            actor_list = actor_list.unsqueeze(1)

        return actor_list

class QNetwork(nn.Module):
    def __init__(self, action_dim ,state_dim):
        super().__init__()
        # 计算输入维度：obs_dim + action_dim
        obs_dim = np.array(state_dim).prod()
        act_dim = np.prod(action_dim)

        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.ln1 = nn.LayerNorm(256)  # LayerNorm 1
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)  # LayerNorm 2
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        # 拼接观测和动作
        x = torch.cat([x, a], dim=1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

def prepare_state(latest_scan, distance, cos, sin, collision, goal, action,state_dim):
    # update the returned data from ROS into a form used for learning in the current model
    latest_scan = np.array(latest_scan)

    inf_mask = np.isinf(latest_scan)
    latest_scan[inf_mask] = 7.0

    max_bins = state_dim - 5
    bin_size = int(np.ceil(len(latest_scan) / max_bins))

    # Initialize the list to store the minimum values of each bin
    min_values = []

    # Loop through the data and create bins
    for i in range(0, len(latest_scan), bin_size):
        # Get the current bin
        bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
        # Find the minimum value in the current bin and append it to the min_values list
        min_values.append(min(bin))
    state = min_values + [distance, cos, sin] + [action[0], action[1]]

    assert len(state) == state_dim
    terminal = 1 if collision or goal else 0

    return state, terminal

def get_action(obs, add_noise,action_dim,max_action,act, qf1, qf2):
    with torch.no_grad():
        if add_noise:
            return (
                act(obs, qf1, qf2).cpu().numpy() + np.random.normal(0, 0.2, size=action_dim)
            ).clip(-max_action, max_action)
        else:
            return act(obs, qf1, qf2).cpu().numpy()

def quaternion_to_euler_tensor(quat):
    """
    将四元数张量转换为欧拉角张量 (XYZ 旋转顺序)

    参数:
        quat: 形状为 (..., 4) 的张量，表示 (w, x, y, z)

    返回:
        euler_angles: 形状为 (..., 3) 的张量，表示 (roll, pitch, yaw)
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # 计算欧拉角（XYZ 旋转顺序）
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    pitch = torch.asin(2 * (w * y - z * x))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))

    return torch.stack((roll, pitch, yaw), dim=-1)


class CarDeploy:
    def __init__(self, ):
        self.actions = torch.zeros(1, 2, dtype=torch.float)
        self.sin = torch.zeros(1, 1, dtype=torch.float)
        self.cos = torch.zeros(1, 1, dtype=torch.float)
        self.distance = torch.zeros(1, 1, dtype=torch.float)
        self.ray2d_ = torch.zeros(1, 10, dtype=torch.float)

    def get_data(self,
                 base_ang_vel,  # 车身三轴角速度
                 base_quat,  # 车身姿态四元数
                 car_pos,  # 车身位置
                 position_targets,  # 目标位置
                 _target_heading,  # 目标朝向
                 dof_pos,  # 四电机位置
                 dof_vel,  # 四电机速度  改成两电机速度
                 actions,  # 模型算出的四电机速度
                 dt,  # 计时
                 ray2d_, ):  # 激光雷达数据
        self.base_ang_vel = base_ang_vel
        self.base_quat = base_quat
        self.car_pos = car_pos
        self.position_targets = position_targets
        self._target_heading = _target_heading
        self.dof_pos = dof_pos
        self.dof_vel = dof_vel
        self.actions = actions
        self.dt = dt
        self.ray2d_ = ray2d_
        #self.ray2d_ = 5*torch.ones(1,10)

    def get_dist_sincos(self):
        euler = quaternion_to_euler_tensor(self.base_quat)
        pose = torch.tensor([torch.cos(euler[:, 2]), torch.sin(euler[:, 2])])
        goal = torch.cat((self.position_targets[:, 0] - self.car_pos[:, 0], self.position_targets[:, 1] - self.car_pos[:, 1]),dim=-1)
        self.distance = torch.norm(goal)
        self.cos, self.sin = self.cossin(pose, goal)

    @staticmethod
    def cossin(vec1, vec2):
        vec1 = vec1 / torch.norm(vec1)
        vec2 = vec2 / torch.norm(vec2)
        cos = torch.dot(vec1, vec2)
        sin = torch.det(torch.stack([vec1, vec2]))  # 计算 2D 矩阵的行列式
        return cos, sin

    def compute_observations(self):
        """ Computes observations
        """
        self.get_dist_sincos()
        self.obs_buf = torch.cat((
            self.ray2d_,
            self.distance.unsqueeze(-1).unsqueeze(-1),
            self.cos.unsqueeze(-1).unsqueeze(-1),
            self.sin.unsqueeze(-1).unsqueeze(-1),
            self.actions  # 18:22
        ), dim=-1)  # append ray2d obs after this, 50:
        # print(self.timer_left.unsqueeze(1))
        # add perceptive inputs if not blind

# dirs = glob.glob(f"/home/ehr/DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2/models/TD3/TD3_actor.pth")
# print(dirs)
# policy = Actor(25,2)
# policy.load_state_dict(torch.load(dirs[0], map_location=torch.device('cpu')))
dirs = glob.glob(f"/home/ehr/DRL-Robot-Navigation-ROS2/runs/0317093201__10__alpha10__nodes10/models/train_kd3rl_0000690000.pt")
print(dirs)
w = torch.load(dirs[0], weights_only=False, map_location=torch.device('cpu'))
policy = ActorDiffusion(2,15,1,-1)
qf1 = QNetwork(2,15)
qf2 = QNetwork(2,15)
policy.load_state_dict(w["actor"])
qf1.load_state_dict(w['qf1'])
qf2.load_state_dict(w['qf2'])
# car = CarDeploy()
# actions = torch.zeros([1, 2], dtype=torch.float32)  # 初始化模型输出  actions改为二维
#
# def replace_nan_with_value(data, value=6.0):
#     # Replace NaN values in the input data with the specified value
#     return [value if math.isnan(x) else x for x in data]
#
# base_ang_vel=torch.zeros([1,3])  # 车身三轴角速度
# base_quat=torch.zeros([1,4])  # 车身姿态四元数
# car_pos=torch.zeros([1,3])  # 车身位置
# position_targets=torch.ones([1,3])  # 目标位置
# _target_heading=torch.zeros([1])  # 目标朝向
# dof_pos=torch.zeros([1,4])  # 四电机位置
# dof_vel=torch.zeros([1,4])  # 四电机速度
# ray2d_=torch.zeros([1,20])  # 激光雷达数据  2.585
# dt = 0
# base_quat[:,0] = 1
# position_targets[:,1]=0
# position_targets[:,2]=0
# while True:
#     car.get_data(
#                     base_ang_vel,      # 车身三轴角速度
#                     base_quat,         # 车身姿态四元数
#                     car_pos,           # 车身位置
#                     position_targets,  # 目标位置
#                     _target_heading,   # 目标朝向
#                     dof_pos,           # 四电机位置
#                     dof_vel,           # 四电机速度
#                     actions,           # 模型算出的四电机速度
#                     dt,
#                     ray2d_)
#     car.compute_observations()
#
#     actions=policy(car.obs_buf)  #线速度，角速度
#     print(car.obs_buf)
ros = ROS_env()  # instantiate ROS environment
latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
        lin_velocity=0.0, ang_velocity=0.0
    )  # get the initial step state
#
# model = TD3(
#         state_dim=25,
#         action_dim=2,
#         max_action=1,
#         device='cuda:0',
#         save_every=1000,
#         load_model=True,
#     )  # instantiate a model

while True:
    state, terminal = prepare_state(
        latest_scan, distance, cos, sin, collision, goal, a, 15
    )  # get state a state representation from returned data from the environment
    # for i in range(10):
    #     state[i]=5
    #action = get_action(state, True,2,1,act, qf1, qf2)  # get an action from the model
    state = torch.tensor([state],dtype=torch.float)
    action = policy(state,qf1,qf2)
    a_in = [
        (action[0][0] + 1) / 2,
        action[0][1],
    ]  # clip linear velocity to [0, 0.5] m/s range

    latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
        lin_velocity=a_in[0], ang_velocity=a_in[1]
    )  # get data from the environment
