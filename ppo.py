import gymnasium as gym
import cv2
import wandb
from argparse import ArgumentParser
from distutils.util import strtobool
from agent import Agent
import torch
from torch import optim
from torch.distributions.categorical import Categorical
import time


parser = ArgumentParser()
parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                    help="the learning rate of the optimizer")
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                    help="if toggled, this experiment will be tracked with Weights and Biases")
parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
                    help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None,
                    help="the entity (team) of wandb's project")
parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                    help="weather to capture videos of the agent performances (check out `videos` folder)")

# Algorithm specific arguments
parser.add_argument("--num-envs", type=int, default=4,
                    help="the number of parallel game environments")
parser.add_argument("--num-steps", type=int, default=128,
                    help="the number of steps to run in each environment per policy rollout")
parser.add_argument("--total-timesteps", type=int, default=25000,
                    help="total timesteps of the experiments")

args = parser.parse_args()

batch_size = args.num_envs*args.num_steps

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_env(gym_id, idx, capture_video, seed):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0 and capture_video:
            env = gym.wrappers.RecordVideo(
                env, 'videos',
                episode_trigger=lambda t: t % 1000 == 0)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


envs = gym.vector.SyncVectorEnv(
    [make_env("CartPole-v1", i, args.capture_video, 0) for i in range(args.num_envs)])

agent = Agent(envs).to(device)
print(agent)

optimizer = optim.Adam(agent.parameters(), args.learning_rate, eps=1e-5)

obs = torch.zeros((args.num_steps, args.num_envs) +
                  envs.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) +
                      envs.single_action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

global_step = 0
start_time = time.time()
next_obs = torch.Tensor(envs.reset()[0]).to(device)
next_done = torch.zeros(args.num_envs).to(device)

num_updates = args.total_timesteps // batch_size
print(num_updates)

print(agent.get_value(next_obs))
