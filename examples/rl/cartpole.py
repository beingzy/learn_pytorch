import math
import random

from torch._C import memory_format

import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple
from itertools import count
from PIL import Image 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from .components import ReplayMemory, Transition
from .learner import DQN
from .utils import get_cart_location, get_screen


env = gym.make('CartPole-V0').unwarpped
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is available and to use it
device = torch.device("cuda" if torch.cuda.is_available() else  "cpu")

resize = T.compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),
    T.ToTensor()])

env.reset()
plt.figure()
plt.imshow(
    get_screen(env, resize, device).cpu().squeeze(0).permute(1, 2, 0).numpy(),
    interpolation='none')
plt.title('Example extracted screen')
plt.show()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

init_screen = get_screen(env, resize, device)
_, _, screen_height, screen_width = init_screen.shape

n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]], 
            device=device,
            dtype=torch.long)


episode_durations = []


def plot_durations():
    global episode_durations
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
   
    plt.figure()
    plt.clf()
    plt.title("Training ...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    """
    """
    if len(memory) < BATCH_SIZE:
        return None

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(
        map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool)
    non_final_next_states = torch.cat([
        s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # compute Q(s_t, a) - the model computes Q(s_t), then we select
    # the columns of actions taken. There are the actions which would
    # have been taken for each batch steate according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # compute V(s_{t+1}) for all next states.
    # expected values of actions for `non_final_states` are computed based
    # on the "older" `target_net`; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such taht we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states)
        .max(1)[0]
        .detach())
    # compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, 
        expected_state_action_values.unsqueeze(1))
    
    # optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50
for i_spisode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen(env, resize, device)
    current_screen = get_screen(env, resize, device)
    state = current_screen - last_screen
    for t in count():
        # select and performan action
        action = select_action(state)