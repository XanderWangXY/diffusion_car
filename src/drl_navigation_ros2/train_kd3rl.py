from pathlib import Path

from TD3.TD3 import TD3
from SAC.SAC import SAC
from ros_python import ROS_env
from replay_buffer import ReplayBuffer
import torch
import numpy as np
from utils import record_eval_positions
from pretrain_utils import Pretraining

import os
import random
import time
from dataclasses import dataclass
import os
import re
import inspect
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from diffusion import Diffusion
from typing import Optional,Callable
from utils_diffusion import DistributionMetrics_Beta
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR
import json
import tyro
from dataclasses import asdict, dataclass, fields
from types import SimpleNamespace
from typing import Optional

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    config_path: str = r"/home/ehr/DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2/config/Ant.json"
    run_name: Optional[str] = None

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
    def __init__(self, action_dim ,state_dim ,max_action,min_action):
        super().__init__()
        condition_dim = np.array(state_dim).prod()
        self.actor_dim = np.array(action_dim).prod()
        # self.actor = Diffusion(condition_dim, out_dim=self.actor_dim, predict_epsilon=True, n_timesteps=5).to(device)
        self.entropy_type = args.entropy_type
        self.eval_sample = args.eval_sample
        self.actor_noise = args.is_actor_noise
        self.eval_noise = args.is_eval_noise
        self.behavior_sample = args.behavior_sample
        self.num_nodes = args.num_nodes
        self.actor = Diffusion(condition_dim, out_dim=self.actor_dim, predict_epsilon=False, n_timesteps=args.diff_step).to(device)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((max_action - min_action) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((max_action + min_action) / 2.0, dtype=torch.float32)
        )
        self.kde = DistributionMetrics_Beta(args.batch_size, args.num_nodes, self.actor_dim, bandwidth=args.bandwidth, scale=args.kl_scale)


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
            noise_std = self.action_scale * args.exploration_noise
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
            noise_std = self.action_scale * args.exploration_noise
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
            noise_std = self.action_scale * args.exploration_noise
            noise = torch.randn_like(x) * noise_std
            x = x + noise
        x = torch.clamp(x, min=-1.0, max=1.0)
        # actor_list =  x * self.action_scale + self.action_bias
        actor_list = x
        if k_repeat == 1:
            actor_list = actor_list.unsqueeze(1)

        return actor_list


# def evaluate(
#     model_path: str,
#     env_id: str,
#     eval_episodes: int,
#     run_name: str,
#     device: torch.device = torch.device("cpu"),
#     capture_video: bool = False,
#     exploration_noise: float = 0.1,
# ):
#     envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
#     actor = Actor(envs).to(device)
#     qf1 = QNetwork(envs).to(device)
#     qf2 = QNetwork(envs).to(device)
#     actor_params, qf1_params, qf2_params = torch.load(model_path, map_location=device)
#     actor.load_state_dict(actor_params)
#     actor.eval()
#     qf1.load_state_dict(qf1_params)
#     qf2.load_state_dict(qf2_params)
#     qf1.eval()
#     qf2.eval()
#     # note: qf1 and qf2 are not used in this script
#     obs, _ = envs.reset()
#     episodic_returns = []
#     while len(episodic_returns) < eval_episodes:
#         with torch.no_grad():
#             actions = actor.get_max_action(torch.Tensor(obs).to(device), qf1, qf2).cpu().numpy()
#             # actions += torch.normal(0, actor.action_scale * exploration_noise)
#             # actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
#         step_actions = actions * action_scale + action_bias
#         next_obs, _, _, _, infos = envs.step(step_actions)
#         if "final_info" in infos:
#             for info in infos["final_info"]:
#                 if "episode" not in info:
#                     continue
#                 print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
#                 episodic_returns += [info["episode"]["r"]]
#         obs = next_obs
#
#     return np.mean(episodic_returns)

def lr_lambda(step):
    fraction = step / float(1e4)
    # ratio = lr_end / lr_start
    # factor = ratio^fraction
    return (5e-5 / args.act_learning_rate) ** fraction

def lr_lambda_q(step):
    fraction = step / float(1e4)
    # ratio = lr_end / lr_start
    # factor = ratio^fraction
    return (5e-5 / args.learning_rate) ** fraction

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
        min_values.append(np.mean(bin))
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

if __name__ == "__main__":
    """Main training function"""
    action_dim = 2  # number of actions produced by the model
    state_dim = 25  # number of input values in the neural network (vector length of state input)
    max_action = 1  # maximum absolute value of output actions
    min_action = -1
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    nr_eval_episodes = 10  # how many episodes to use to run evaluation
    max_epochs = 100  # max number of epochs
    epoch = 0  # starting epoch number
    episodes_per_epoch = 70  # how many episodes to run in single epoch
    episode = 0  # starting episode number
    train_every_n = 2  # train and update network parameters every n episodes
    training_iterations = 500  # how many batches to use for single training cycle
    batch_size = 40  # batch size for each training iteration
    max_steps = 300  # maximum number of steps in single episode
    steps = 0  # starting step number
    load_saved_buffer = True  # whether to load experiences from assets/data.yml
    pretrain = True  # whether to use the loaded experiences to pre-train the model (load_saved_buffer must be True)
    pretraining_iterations = (
        50  # number of training iterations to run during pre-training
    )
    save_every = 100  # save the model every n training cycles
    infos=[{}]

    # model = TD3(
    #     state_dim=state_dim,
    #     action_dim=action_dim,
    #     max_action=max_action,
    #     device=device,
    #     save_every=save_every,
    #     load_model=False,
    # )  # instantiate a model
    if load_saved_buffer:
        # pretraining = Pretraining(
        #     file_names=["src/drl_navigation_ros2/assets/data.yml"],
        #     model=model,
        #     replay_buffer=ReplayBuffer(buffer_size=5e3, random_seed=42),
        #     reward_function=ros.get_reward,
        # )  # instantiate pre-trainind
        # replay_buffer = (
        #     pretraining.load_buffer()
        # )  # fill buffer with experiences from the data.yml file
        # if pretrain:
        #     pretraining.train(
        #         pretraining_iterations=pretraining_iterations,
        #         replay_buffer=replay_buffer,
        #         iterations=training_iterations,
        #         batch_size=batch_size,
        #     )  # run pre-training
        print('pass')
    else:
        # replay_buffer = ReplayBuffer(
        #     buffer_size=5e3, random_seed=42
        # )  # if not experiences are loaded, instantiate an empty buffer
        print('pass')

    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    t_args = tyro.cli(Args)
    with open(t_args.config_path, "r") as f:
        args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    args.exp_name = t_args.exp_name

    if args.run_name != None:
        run_name = args.run_name
    else:
        run_name = f"{time.strftime('%m%d%H%M%S')}__{args.seed}__alpha{args.alpha_smooth}__nodes{args.num_nodes}"

    os.makedirs(f"runs/{run_name}", exist_ok=True)
    # Save args as JSON under the run_name
    run_filename = f"runs/{run_name}/{args.env_id}.json"
    with open(run_filename, "w") as f:
        json.dump(vars(args), f, indent=4)

    # Print confirmation
    print(f"Arguments saved to {run_filename}")
    # 创建一个奖励记录文件
    reward_file_path = f"runs/{run_name}/rewards.txt"
    reward_file = open(reward_file_path, "w")
    reward_file.write("global_step,episodic_return\n")

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    #envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    #eval_env = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    #assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(action_dim,state_dim,max_action,min_action).to(device)
    qf1 = QNetwork(action_dim ,state_dim).to(device)
    qf2 = QNetwork(action_dim ,state_dim).to(device)
    qf1_target = QNetwork(action_dim ,state_dim).to(device)
    qf2_target = QNetwork(action_dim ,state_dim).to(device)
    target_actor = Actor(action_dim,state_dim,max_action,min_action).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())

    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.act_learning_rate)
    actor_scheduler = LambdaLR(actor_optimizer, lr_lambda=lr_lambda)
    q_scheduler = LambdaLR(q_optimizer, lr_lambda=lr_lambda_q)
    # 初始化图平滑度自动调节
    if args.autotune_smoothness:

        target_smoothness = actor.kde.log_uniform_pdf
        target_smoothness = torch.tensor(target_smoothness, device=device, dtype=torch.float32)
        alpha_smooth = args.alpha_smooth

    else:
        target_smoothness = actor.kde.log_uniform_pdf
        # 如果不自动调节，使用固定的 alpha_smooth
        alpha_smooth = args.alpha_smooth
    action_scale = actor.action_scale.detach().cpu().numpy()
    action_bias = actor.action_bias.detach().cpu().numpy()

    # envs.single_observation_space.dtype = np.float32
    observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    rb = ReplayBuffer(
        args.buffer_size,
        observation_space,
        action_space,
        device,
        handle_timeout_termination=False,
    )

    ros = ROS_env()  # instantiate ROS environment
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes
    )  # save scenarios that will be used for evaluation

    start_time = time.time()

    latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
        lin_velocity=0.0, ang_velocity=0.0
    )  # get the initial step state

    for global_step in range(args.total_timesteps+1):
        state, terminal = prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a, state_dim
        )  # get state a state representation from returned data from the environment
        # if global_step % args.eval_interval == 0:
        #     model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        #     torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
        #     with torch.no_grad():
        #         # eval_episodic_return = evaluate_policy(actor, eval_env, qf1=qf1, qf2=qf2, eval_episodes=10, device=device)
        #         eval_episodic_return = evaluate(model_path=model_path,run_name=run_name,eval_episodes=10,device=device,env_id=args.env_id)
        #         print(f"global_step={global_step}, eval_episodic_return={eval_episodic_return}")
        #         writer.add_scalar("eval/episodic_return", eval_episodic_return, global_step)
        #         reward_file.write(f"{global_step},{eval_episodic_return}\n")
        #         reward_file.flush()  # 确保实时写入文件
        # ALGO LOGIC: put action logic here

        if global_step < args.learning_starts:
            actions = np.array([action_space.sample() ])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(np.asarray([state], dtype=np.float32)).to(device), qf1, qf2).cpu().numpy()
                # actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                # actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        step_actions = actions * action_scale + action_bias
        a_in = [
            (step_actions[0][0] + 1) / 2,
            step_actions[0][1],
        ]  # clip linear velocity to [0, 0.5] m/s range
        latest_scan, distance, cos, sin, collision, goal, a, rewards = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )  # get data from the environment
        next_obs, terminations = prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a, state_dim
        )  # get state a state representation from returned data from the environment

        rewards *= args.reward_scale
        # TRY NOT TO MODIFY: record rewards for plotting purposes

        # if "episode" in infos:
        #     episodic_return = {infos['episode']['r'][0]}
        #     print(f"global_step={global_step}, episodic_return={infos['episode']['r'][0]}")
        #     writer.add_scalar("charts/episodic_return", infos["episode"]["r"][0], global_step)
        #     writer.add_scalar("charts/episodic_length", infos["episode"]["l"][0], global_step)
        #     # 记录到奖励文件
        #     reward_file.write(f"{global_step},{episodic_return}\n")
        #     reward_file.flush()  # 确保实时写入文件
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        #real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc and "final_observation" in infos:
        #         real_next_obs[idx] = infos["final_observation"][idx]

        obs=np.array(state)
        real_next_obs=np.array(next_obs.copy())
        terminations=np.array(terminations)
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        if (
            terminal or steps == max_steps
        ):  # reset environment of terminal stat ereached, or max_steps were taken
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():

                next_state_actions = target_actor.get_K_actions(data.next_observations, k_repeat=args.k_critic, is_graph=False)
                expanded_obs = data.next_observations.unsqueeze(1).repeat(1, args.k_critic, 1).view(-1, data.next_observations.size(-1))
                reshaped_acts = next_state_actions.reshape(args.batch_size * args.k_critic, -1)
                if args.is_Q_noise:
                    clipped_noise = (torch.randn_like(reshaped_acts, device=device) * args.policy_noise).clamp(
                        -args.noise_clip, args.noise_clip
                    ) * target_actor.action_scale

                    reshaped_acts = (reshaped_acts + clipped_noise).clamp(-1,1)

                qf1_next_target = qf1_target(expanded_obs, reshaped_acts)
                qf2_next_target = qf2_target(expanded_obs, reshaped_acts)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                min_qf_next_target = min_qf_next_target.view(args.batch_size, args.k_critic, 1)
                # min_qf_next_target = min_qf_next_target.mean(dim=1, keepdim=False) - alpha_smooth * graph_smooth.unsqueeze(1)
                min_qf_next_target = min_qf_next_target.mean(dim=1, keepdim=False)

                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            critic_grad_norms = nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()),
                                                         max_norm=2,
                                                         norm_type=2)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                act_list = actor.get_K_actions(data.observations, k_repeat=args.num_nodes)
                expanded_obs = data.observations.unsqueeze(1).repeat(1, args.num_nodes, 1).view(-1,data.observations.size(-1))
                reshaped_acts = act_list.reshape(args.batch_size * args.num_nodes, -1)
                q_1 = qf1(expanded_obs, reshaped_acts)
                q_2 = qf2(expanded_obs, reshaped_acts)
                q_min = torch.min(q_1, q_2)
                q_values1 = q_min.view(args.batch_size, args.num_nodes, 1)
                graph_smooth = actor.calculate_graph_smoothness_einsum(act_list, q_values1)
                average_q_value1 = q_values1.mean(dim=1).mean()  # [batch]
                # adv_value = (q_values1 - q_values1.mean(dim=1, keepdim=True))
                # average_q_value1 = adv_value[adv_value > 0].mean()
                graph_smooth = graph_smooth.mean()
                actor_loss = -average_q_value1 + alpha_smooth*graph_smooth
                #actor_loss = -average_q_value1
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norms = nn.utils.clip_grad_norm_(list(actor.parameters()),
                                                             max_norm=2,
                                                             norm_type=2)
                actor_optimizer.step()

                # 动态调整 target_smoothness

                # # 自适应调整 alpha_smooth（如果启用自动调节）
                if args.autotune_smoothness:
                    with torch.no_grad():
                    # 定义 alpha_smooth 的上下限
                        MAX_ALPHA_SMOOTH = 100.0
                        MIN_ALPHA_SMOOTH = -100.0
                        # 直接使用之前计算的 graph_smooth，并从计算图中分离
                        graph_smooth_detached = graph_smooth.mean().detach()
                        smooth_loss = (graph_smooth_detached - target_smoothness).mean()
                        alpha_smooth = alpha_smooth + args.alpha_learning_rate * smooth_loss
                        alpha_smooth = max(min(alpha_smooth, MAX_ALPHA_SMOOTH), MIN_ALPHA_SMOOTH)


                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 10000 == 0:
                if args.is_auto_sch:
                    actor_scheduler.step()
                if args.is_q_auto_sch:
                    q_scheduler.step()
                current_lr = actor_optimizer.param_groups[0]['lr']
                writer.add_scalar("losses/current_lr", current_lr, global_step)
                current_lr_q = q_optimizer.param_groups[0]['lr']
                writer.add_scalar("losses/current_lr_q", current_lr_q, global_step)
                writer.add_histogram("losses/act_hist", act_list[0, :, 0], global_step)
                writer.add_histogram("losses/q_", q_values1[0].reshape(-1), global_step)
                writer.add_scalar("losses/alpha_smooth", alpha_smooth, global_step)
                writer.add_scalar("losses/average_q_value1", average_q_value1.item(), global_step)
                writer.add_scalar("losses/graph_smooth", graph_smooth.item(), global_step)
                writer.add_scalar("losses/target_smooth", target_smoothness.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                print("..............................................")
                print(f"Epoch {global_step}. Evaluating {len(eval_scenarios)} scenarios")
                avg_reward = 0.0
                col = 0
                gl = 0
                for scenario in eval_scenarios:
                    count = 0
                    latest_scan, distance, cos, sin, collision, goal, a, reward = ros.eval(
                        scenario=scenario
                    )
                    while count < max_steps:
                        state, terminal = prepare_state(
                            latest_scan, distance, cos, sin, collision, goal, a,state_dim
                        )
                        if terminal:
                            break
                        action = get_action(torch.Tensor(np.asarray([state], dtype=np.float32)).to(device), False,action_dim,max_action,actor, qf1, qf2)
                        a_in = [(action[0][0] + 1) / 2, action[0][1]]
                        latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
                            lin_velocity=a_in[0], ang_velocity=a_in[1]
                        )
                        avg_reward += reward
                        count += 1
                        col += collision
                        gl += goal
                avg_reward /= len(eval_scenarios)
                avg_col = col / len(eval_scenarios)
                avg_goal = gl / len(eval_scenarios)
                print(f"Average Reward: {avg_reward}")
                print(f"Average Collision rate: {avg_col}")
                print(f"Average Goal rate: {avg_goal}")
                print("..............................................")
                writer.add_scalar("eval/avg_reward", avg_reward, global_step)
                writer.add_scalar("eval/avg_col", avg_col, global_step)
                writer.add_scalar("eval/avg_goal", avg_goal, global_step)

                # 添加以下代码到if global_step % 10000 == 0:的最后部分

                # 创建保存模型的目录
                model_dir = f"runs/{run_name}/models"
                os.makedirs(model_dir, exist_ok=True)

                # 保存模型，使用步数作为名称，确保不会覆盖之前的模型
                model_path = f"{model_dir}/{args.exp_name}_{global_step:010d}.pt"  # 使用10位数字格式化，便于排序
                print(f"Saving model to {model_path}")
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "qf1": qf1.state_dict(),
                        "qf2": qf2.state_dict(),
                        "target_actor": target_actor.state_dict(),
                        "qf1_target": qf1_target.state_dict(),
                        "qf2_target": qf2_target.state_dict(),
                        "actor_optimizer": actor_optimizer.state_dict(),
                        "q_optimizer": q_optimizer.state_dict(),
                        "actor_scheduler": actor_scheduler.state_dict() if args.is_auto_sch else None,
                        "q_scheduler": q_scheduler.state_dict() if args.is_q_auto_sch else None,
                        "alpha_smooth": alpha_smooth,
                        "global_step": global_step,
                        "avg_reward": avg_reward,  # 同时保存评估指标
                        "avg_col": avg_col,
                        "avg_goal": avg_goal
                    },
                    model_path
                )
                print(f"Model saved successfully at step {global_step}")

                # 可选：记录所有保存的模型文件到一个列表文件中
                model_list_path = f"runs/{run_name}/saved_models.txt"
                with open(model_list_path, "a") as f:
                    f.write(f"{global_step},{model_path},{avg_reward},{avg_col},{avg_goal}\n")
        # Evaluate policy every 10,000 steps