import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import time
import math

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
        self.ray2d_ = torch.zeros(1, 20, dtype=torch.float)

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
        self.ray2d_ = 5*torch.ones(1,20)

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

dirs = glob.glob(f"/home/ehr/DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2/models/TD3/TD3_actor.pth")
print(dirs)
policy = Actor(25,2)
policy.load_state_dict(torch.load(dirs[0], map_location=torch.device('cpu')))
car = CarDeploy()
actions = torch.zeros([1, 2], dtype=torch.float32)  # 初始化模型输出  actions改为二维

def replace_nan_with_value(data, value=6.0):
    # Replace NaN values in the input data with the specified value
    return [value if math.isnan(x) else x for x in data]

base_ang_vel=torch.zeros([1,3])  # 车身三轴角速度
base_quat=torch.zeros([1,4])  # 车身姿态四元数
car_pos=torch.zeros([1,3])  # 车身位置
position_targets=torch.ones([1,3])  # 目标位置
_target_heading=torch.zeros([1])  # 目标朝向
dof_pos=torch.zeros([1,4])  # 四电机位置
dof_vel=torch.zeros([1,4])  # 四电机速度
ray2d_=5*torch.ones([1,20])  # 激光雷达数据  2.585
dt = 0
base_quat[:,0] = 1
position_targets[:,1]=0
position_targets[:,2]=0
while True:
    car.get_data(
                    base_ang_vel,      # 车身三轴角速度
                    base_quat,         # 车身姿态四元数
                    car_pos,           # 车身位置
                    position_targets,  # 目标位置
                    _target_heading,   # 目标朝向
                    dof_pos,           # 四电机位置
                    dof_vel,           # 四电机速度
                    actions,           # 模型算出的四电机速度
                    dt,
                    ray2d_)
    car.compute_observations()

    actions=policy(car.obs_buf)  #线速度，角速度
    print(car.obs_buf)