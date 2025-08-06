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
from meerl_diffusion import Diffusion
from typing import Optional, Callable

from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR
import json
import tyro
from dataclasses import asdict, dataclass, fields
from types import SimpleNamespace
from typing import Optional
import datetime
import csv
import pandas as pd
from meerl_utils_diffusion import MEECalculator_1D, MEECalculator, MEECalculator_Huber


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    config_path: str = r"/home/ehr/DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2/config_mee/Humanoid.json"
    # config_path: str = r"E:\cleanrl\MEERL\config\Humanoid.json"
    run_name: Optional[str] = None


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
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
    def __init__(self, action_dim ,state_dim ,action_space, args):
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
            actions = act(obs, qf1, qf2)
            return actions[0][0].cpu().numpy()

if __name__ == "__main__":
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
    if args.run_name is None:
        run_name = f"{args.env_id}/{args.exp_name}/{args.seed}/{datetime.datetime.now().strftime('%m%d_%H%M%S')}"
    else:
        run_name = args.run_name

    # 创建txt文件用于记录
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "rewards.txt")

    # 创建并写入表头
    with open(log_path, 'w') as f:
        f.write("global_step,episodic_return\n")

    # 打开文件用于追加写入
    log_file = open(log_path, 'a')
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    # Save args as JSON under the run_name
    run_filename = f"runs/{run_name}/{args.env_id}.json"
    with open(run_filename, "w") as f:
        json.dump(vars(args), f, indent=4)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    actor = Actor(action_dim,state_dim,action_space, args).to(device)
    qf1 = QNetwork(action_dim ,state_dim).to(device)
    qf2 = QNetwork(action_dim ,state_dim).to(device)
    qf1_target = QNetwork(action_dim ,state_dim).to(device)
    qf2_target = QNetwork(action_dim ,state_dim).to(device)
    target_actor = Actor(action_dim,state_dim,action_space, args).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.act_learning_rate)


    rb = ReplayBuffer(
        args.buffer_size,
        observation_space,
        action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    ros = ROS_env()  # instantiate ROS environment
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes
    )  # save scenarios that will be used for evaluation

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
        lin_velocity=0.0, ang_velocity=0.0
    )  # get the initial step state
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        state, terminal = prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a, state_dim
        )  # get state a state representation from returned data from the environment
        if global_step < args.learning_starts:
            actions = np.array([action_space.sample()])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(np.asarray([state], dtype=np.float32)).to(device),Qf1=qf1, Qf2=qf2, mode="sample", k_repeat=args.behavior_sample, use_prob_sampling=args.use_prob_sampling).cpu().numpy()
                if args.is_actor_noise:
                    actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                #actions = actions.cpu().numpy().clip(min_action, max_action)

        # TRY NOT TO MODIFY: execute the game and log data.
        #step_actions = actions * action_scale + action_bias
        step_actions = actions
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

        # 记录每个步骤的奖励
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    # 实时写入txt文件
                    log_file.write(f"{global_step},{info['episode']['r']}\n")
                    log_file.flush()  # 确保立即写入磁盘
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
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
                if args.is_Q_noise:
                    clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                        -args.noise_clip, args.noise_clip
                    ) * target_actor.action_scale

                    next_state_actions = (target_actor(data.next_observations, Qf1=qf1, Qf2=qf2, mode="sample", k_repeat=args.target_sample, use_prob_sampling=args.Q_use_prob_sampling) + clipped_noise).clamp(
                        min_action, max_action
                    )
                else:
                    next_state_actions = target_actor(data.next_observations, Qf1=qf1, Qf2=qf2, mode="sample", k_repeat=args.target_sample, use_prob_sampling=args.Q_use_prob_sampling)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            nn.utils.clip_grad_norm_(qf1.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(qf2.parameters(), max_norm=1.0)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                batch_size = data.observations.shape[0]
                states = data.observations
                num_nodes = args.num_nodes
                # with torch.no_grad():
                actions, best_action, qadv, q_values = actor(states, Qf1=qf1, Qf2=qf2, mode="run", k_repeat=num_nodes)  # [batch_size, num_nodes, action_dim]
                # actions_reshape = actions.reshape(batch_size, num_nodes, -1)
                
                    # rand_states = states.unsqueeze(0).expand(10, -1, -1).contiguous().view(batch_size*10, -1)
                    # rand_policy_actions = torch.empty(batch_size * 10, actions.shape[-1], device=device).uniform_(envs.single_action_space.low[0], envs.single_action_space.high[0])
                    # rand_q = qadv.unsqueeze(0).expand(10, -1, -1).contiguous().view(batch_size*10, -1) * 0.02
                    # best_actions_with_rand = torch.cat([best_action, rand_policy_actions], dim=0)
                    # states_with_rand = torch.cat([states, rand_states], dim=0)
                    # qadv_with_rand = torch.cat([qadv, rand_q], dim=0)
                # actor_loss = (args.alpha_smooth * mee_per_batch - q_values_mean).mean()
                # actor_loss_diffusion = actor.diffusion.get_loss(x_start=best_actions_with_rand, state=states_with_rand, weight=qadv_with_rand)
                actor_loss_diffusion = actor.diffusion.get_loss(x_start=best_action, state=states, weight=qadv)
                if args.use_entropy:
                    if args.use_huber:
                        mee_per_batch, total_mee, avg_valid_nodes = actor.mee_calculator(q_values, actions.reshape(batch_size, num_nodes, -1), delta=args.delta)
                    else:
                        mee_per_batch, total_mee = actor.mee_calculator(actions.reshape(batch_size, num_nodes, -1))
                    actor_loss = (args.alpha_smooth * total_mee + actor_loss_diffusion)
                else:
                    actor_loss = actor_loss_diffusion
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 10000 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/q_values", q_values.std().item(), global_step)
                if args.use_entropy:
                    writer.add_scalar("losses/mee_per_batch", mee_per_batch.mean().item(), global_step)
                if args.use_huber:
                    writer.add_scalar("losses/avg_valid_nodes", avg_valid_nodes.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
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
                            latest_scan, distance, cos, sin, collision, goal, a, state_dim
                        )
                        if terminal:
                            break
                        action = get_action(torch.Tensor(np.asarray([state], dtype=np.float32)).to(device), False,
                                            action_dim, max_action, actor, qf1, qf2)
                        a_in = [(action[0] + 1) / 2, action[1]]
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
                        # "actor_scheduler": actor_scheduler.state_dict() if args.is_auto_sch else None,
                        # "q_scheduler": q_scheduler.state_dict() if args.is_q_auto_sch else None,
                        # "alpha_smooth": alpha_smooth,
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



    # 关闭文件
    log_file.close()
    writer.close()

