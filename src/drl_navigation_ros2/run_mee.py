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
from meerl_diffusion import Diffusion
from meerl_utils_diffusion import MEECalculator_1D, MEECalculator, MEECalculator_Huber

class ActorDiffusion(nn.Module):
    def __init__(self, action_dim, state_dim, action_space, args):
        super().__init__()
        self.fc1 = nn.Linear(np.array(state_dim).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(action_dim))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        if args.use_huber:
            self.mee_calculator = MEECalculator_Huber(bandwidth=args.bandwidth, device=torch.device('cuda'))
        else:
            self.mee_calculator = MEECalculator(bandwidth=args.bandwidth, device=torch.device('cuda'))
        # 添加diffusion模型
        self.diffusion = Diffusion(
            condition_dim=np.array(state_dim).prod(),
            out_dim=np.prod(action_dim),
            beta_schedule="vp",
            n_timesteps=args.diff_step,
            predict_epsilon=args.predict_epsilon,
            device="cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )
        self.args = args

    def sample_best_actions(self, q_values, actions, batch_size, k_repeat=10, use_prob_sampling=False, temperature=1.0):
        # temperature参数可以控制采样的"温度"，较高的值会使概率分布更平坦，增加随机性；较低的值会使分布更尖锐，更倾向于选择高Q值的动作。
        if use_prob_sampling:
            # 基于概率采样
            q_probs = F.softmax(q_values * temperature, dim=1)
            sampled_idx = torch.multinomial(q_probs.squeeze(-1), 1).squeeze(-1)
            batch_indices = torch.arange(batch_size, device=q_values.device)
            best_action = actions.reshape(batch_size, k_repeat, -1)[batch_indices, sampled_idx]
        else:
            # 最大值采样
            q_idx = q_values.argmax(dim=1)
            batch_indices = torch.arange(batch_size, device=q_values.device)
            best_action = actions.reshape(batch_size, k_repeat, -1)[batch_indices, q_idx.squeeze()]
        return best_action

    def forward(self, x, Qf1, Qf2, k_repeat=1, mode="run", use_prob_sampling=False, temperature=1.0):
        # 使用diffusion采样生成动作
        batch_size = x.shape[0]
        shape = torch.Size([batch_size, np.prod(self.action_scale.shape)])

        if mode == "sample":
            stride = self.args.sample_stride
            # 采样动作
            actions_sample = self.diffusion.sample(
                state=x,
                shape=shape,
                type="ddpm",
                stride=stride,
                K_repeat=k_repeat
            )
            actions = actions_sample.reshape(batch_size * k_repeat, -1)

            # 确保动作在动作空间范围内
            actions = actions.clamp(-1, 1) * self.action_scale + self.action_bias

            # 计算Q值
            expanded_obs = x.repeat_interleave(k_repeat, dim=0)
            qf1_values = Qf1(expanded_obs, actions)
            qf2_values = Qf2(expanded_obs, actions)
            q_values = torch.min(qf1_values, qf2_values).reshape(batch_size, k_repeat, -1)

            # 使用sample_best_actions函数根据选择的采样模式返回动作
            best_action = self.sample_best_actions(
                q_values=q_values,
                actions=actions,
                batch_size=batch_size,
                k_repeat=k_repeat,
                use_prob_sampling=use_prob_sampling,
                temperature=temperature
            )

            return best_action
        elif mode == "run":
            stride = self.args.stride
            # 采样动作
            actions_sample = self.diffusion.sample(
                state=x,
                shape=shape,
                type="ddpm",
                stride=stride,
                K_repeat=k_repeat
            )
            actions = actions_sample.reshape(batch_size * k_repeat, -1)

            # 确保动作在动作空间范围内
            actions = actions.clamp(-1, 1) * self.action_scale + self.action_bias

            # 计算Q值
            expanded_obs = x.repeat_interleave(k_repeat, dim=0)
            qf1_values = Qf1(expanded_obs, actions)
            qf2_values = Qf2(expanded_obs, actions)

            with torch.no_grad():
                q_values = torch.min(qf1_values, qf2_values).reshape(batch_size, k_repeat, -1)
                q_values_mean = q_values.mean(dim=1)
                q_idx = q_values.argmax(dim=1)
                batch_indices = torch.arange(batch_size, device=q_values.device)
                best_action = actions.reshape(batch_size, k_repeat, -1)[batch_indices, q_idx.squeeze()]
                best_q = q_values.gather(dim=1, index=q_idx.unsqueeze(1)).squeeze(1)
                adv = (best_q - q_values_mean).detach()
                adv = torch.clamp(adv, min=0) + 1
            return actions, best_action, adv.detach(), q_values.detach()

class QNetwork(nn.Module):
    def __init__(self, action_dim ,state_dim):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(state_dim).prod() + np.prod(action_dim),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
