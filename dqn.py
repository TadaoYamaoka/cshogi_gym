import gym
import cshogi.gym_shogi
from cshogi import *
from cshogi import KIF

import os
import datetime
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 入力特徴量：駒の種類×手番 + 持ち駒の種類×手番
FEATURES_NUM = 14 * 2 + 7 * 2

# 移動の定数
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
] = range(20)

# 指し手を表すラベルの数
MAX_MOVE_LABEL_NUM = len(MOVE_DIRECTION) + 7 # 7はhand piece

def make_position_features(board, features):
    # piece
    board.piece_planes_rotate(features)
    pieces_in_hand = board.pieces_in_hand
    if board.turn == WHITE:
        # 白の場合、反転
        pieces_in_hand = (pieces_in_hand[1], pieces_in_hand[0])
    for c, hands in enumerate(pieces_in_hand):
        for hp, num in enumerate(hands):
            if hp == HPAWN:
                max_hp_num = 8
            elif hp == HBISHOP or hp == HROOK:
                max_hp_num = 2
            else:
                max_hp_num = 4

            features[28 + c * 7 + hp].fill(num / max_hp_num)

def make_output_label(move):
    # 移動先座標
    to_sq = move_to(move)
    # 移動方向((8方向+桂馬2方向)×成2+持ち駒7種類)
    if not move_is_drop(move):
        from_sq = move_from(move)

        to_file, to_rank = divmod(to_sq, 9)
        from_file, from_rank = divmod(from_sq, 9)

        if to_file == from_file:
            if to_rank > from_rank:
                dir = UP
            else:
                dir = DOWN
        elif to_rank == from_rank:
            if to_file > from_file:
                dir = LEFT
            else:
                dir = RIGHT
        elif to_rank - from_rank == 2 and abs(to_file - to_file):
            if to_file > from_file:
                dir = UP2_LEFT
            else:
                dir = UP2_RIGHT
        elif to_file > from_file:
            if to_rank > from_rank:
                dir = UP_LEFT
            else:
                dir = DOWN_LEFT
        else:
            if to_rank > from_rank:
                dir = UP_RIGHT
            else:
                dir = DOWN_RIGHT

        if move_is_promotion(move):
            dir += 10
    else:
        dir = 20 + move_drop_hand_piece(move)

    return 9 * 9 * dir + to_sq


env = gym.make('Shogi-v0').unwrapped

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Replay Memory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# DQN

k = 64
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(FEATURES_NUM, k, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(k)
        self.conv2 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(k)
        self.conv3 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(k)
        self.head = nn.Linear(k * 9 * 9, MAX_MOVE_LABEL_NUM * 9 * 9)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def get_state(env):
    features = np.zeros((1, FEATURES_NUM, 9, 9), dtype=np.float32)
    make_position_features(env.board, features[0])
    state = torch.from_numpy(features[:1]).to(device)
    return state

######################################################################
# Training

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
OPTIMIZE_PER_EPISODES = 5
TARGET_UPDATE = 10

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-3)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(board, state):
    global steps_done

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    legal_labels = []
    legal_moves = []
    for move in board.legal_moves:
        legal_labels.append(make_output_label(move))
        legal_moves.append(move)

    if sample > eps_threshold:
        with torch.no_grad():
            q = policy_net(state)
            select = q[0, legal_labels].argmax()
    else:
        select = random.randrange(len(legal_labels))
    return legal_moves[select], torch.tensor([[legal_labels[select]]], device=device, dtype=torch.long)


episode_durations = []


######################################################################
# Training loop

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    print(f"loss = {loss.item()}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


######################################################################
# main training loop

# 棋譜保存用
os.makedirs('kifu', exist_ok=True)
kif = KIF.Exporter()

num_episodes = 1000
transitions_per_episode = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    state = get_state(env)
    #env.render('sfen')
    kif.open(os.path.join('kifu', datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.kif'))
    kif.header(['dqn', 'dqn'])
    
    for t in count():
        # Select and perform an action
        move, action = select_action(env.board, state)
        _, reward, done, _ = env.step(move)
        reward = torch.tensor([reward], device=device)
        #env.render('sfen')

        # Observe new state
        if not done:
            next_state = get_state(env)
        else:
            next_state = None

        # 棋譜出力
        kif.move(move)
        if done:
            if env.is_draw == cshogi.REPETITION_DRAW:
                kif.end('sennichite')
            elif env.is_draw == cshogi.REPETITION_WIN:
                kif.end('illegal_win')
            elif env.is_draw == cshogi.REPETITION_LOSE:
                kif.end('illegal_lose')
            else:
                kif.move('resign')

        # 終局まで報酬がわからないため遷移を保存しておく
        transitions_per_episode.append((state, action, next_state))

        # Move to the next state
        state = next_state

        if done:
            # 手番を持っている側からみた報酬に変換
            reward = reward if t % 2 == 0 else -reward
            for transition in transitions_per_episode:
                # Store the transition in memory
                memory.push(*transition, reward)
                reward = -reward
            transitions_per_episode.clear()
            kif.close()
            break

    if i_episode % OPTIMIZE_PER_EPISODES == OPTIMIZE_PER_EPISODES - 1:
        # Perform several episodes of the optimization (on the target network)
        optimize_model()

        # Update the target network, copying all weights and biases in DQN
        if i_episode // OPTIMIZE_PER_EPISODES % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()