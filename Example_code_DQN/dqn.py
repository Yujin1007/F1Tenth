import os
import sys
import collections
import random
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Hyperparameters
learning_rate = 0.00005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
train_start = 20000

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

RACETRACK = 'map_easy3'

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(405, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon, memory_size):
        if memory_size < train_start:
            return random.randint(0, 4)
        else:
            out = self.forward(obs)
            coin = random.random()
            if coin < epsilon:
                return random.randint(0,4)
            else:
                return out.argmax().item()
    def action(self, obs):
        out = self.forward(obs)
        return out.argmax().item()

def plot_durations(laptimes):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(laptimes, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 10개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def checkDone(laptimes):
    durations_t = torch.tensor(laptimes, dtype=torch.float)
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        idx = len(means) - 1
        thresh = max(laptimes) * (2 / 3)
        # if means[idx] >= thresh:
        #     plt.figure(2)
        #     plt.clf()
        #     durations_t = torch.tensor(laptimes, dtype=torch.float)
        #     plt.title('Training...')
        #     plt.xlabel('Episode')
        #     plt.ylabel('Duration')
        #     plt.plot(durations_t.numpy())
        #     # 10개의 에피소드 평균을 가져 와서 도표 그리기
        #     if len(durations_t) >= 10:
        #         means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        #         means = torch.cat((torch.zeros(9), means))
        #         plt.plot(means.numpy())
        #     plt.savefig('figure1.png', dpi=300)
        #     return 1
    else:
        return 0

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def preprocess_lidar(ranges):
    eighth = int(len(ranges) / 8)

    return np.array(ranges[eighth:-eighth : 2])

def main():
    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)
    q = Qnet()
    #q.load_state_dict(torch.load("{}\weigths\model_state_dict_easy1_fin.pt".format(current_dir)))
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    # poses = np.array([[0., 0., np.radians(0)]])
    poses = np.array([[0.60070, -0.2753, 1.5707]])

    print_interval = 10
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    speed = 3.0
    laptimes = []

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        obs, r, done, info = env.reset(poses=poses)
        s = preprocess_lidar(obs['scans'][0])
        done = False

        env.render()


        laptime = 0.0

        while not done:
            actions = []

            a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
            steer = (a - 2) * (np.pi / 30)
            if a == 2:
                speed = 5.0
            elif a == 1 or a == 3:
                speed = 4.5
            else:
                speed = 4.0
            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            s_prime = preprocess_lidar(obs['scans'][0])
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100, s_prime, done_mask))
            s = s_prime

            laptime += r
            env.render(mode='human_fast')

            if done:
                laptimes.append(laptime)
                plot_durations(laptimes)
                break

        if memory.size() > train_start:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%"
                  .format(n_epi, laptime/print_interval, memory.size(), epsilon*100))
        if checkDone(laptimes):
            torch.save(q.state_dict(), "{}\weigths\model_state_dict_easy1_new.pt".format(current_dir))
            break

    print('train finish')
    env.close()

def eval():
    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = Qnet()
    q.load_state_dict(torch.load("{}\weigths\model_state_dict_easy1_fin.pt".format(current_dir)))
    poses = np.array([[0., 0., np.radians(90)]])
    speed = 3.0
    for t in range(5):
        obs, r, done, info = env.reset(poses=poses)
        s = preprocess_lidar(obs['scans'][0])

        env.render()
        done = False

        laptime = 0.0

        while not done:
            actions = []

            a = q.action(torch.from_numpy(s).float())
            steer = (a - 2) * (np.pi / 30)
            '''if a == 2:
                speed = 5.0
            elif a == 1 or a == 3:
                speed = 4.5
            else:
                speed = 4.0'''
            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            s_prime = preprocess_lidar(obs['scans'][0])

            s = s_prime

            laptime += r
            env.render(mode='human_fast')

            if done:
                break
    env.close()

if __name__ == '__main__':
    main()
    # eval()
