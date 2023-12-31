import random
from collections import deque
import numpy as np
import gym
import torch

class Qnet(torch.nn.Module):
    def __init__(self, action_size: int):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(4, 128)
        self.fc2 = torch.nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, gamma: float, lr: float, epsilon: float, buffer_size: int, batch_size: int, action_size: int):
        self._gamma = gamma
        self._lr = lr
        self._epsilon = epsilon
        self._action_size = action_size
        self._batch_size = batch_size
        self._replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self._qnet = Qnet(action_size)
        self._qnet_target = Qnet(action_size)
        self._optimizer = torch.optim.Adam(self._qnet.parameters(), lr=lr)

    def sync_qnet(self):
        self._qnet_target.load_state_dict(self._qnet.state_dict())
    
    def get_action(self, state: np.ndarray):
        if np.random.rand() < self._epsilon:
            return np.random.choice(self._action_size)
        else:
            state = torch.tensor(state[np.newaxis, :]).to(torch.float)
            qs = self._qnet(state)
            action = qs.argmax().item()
        return action
    
    def update(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool):
        self._replay_buffer.add(state, action, reward, next_state, done)
        if len(self._replay_buffer) < self._replay_buffer._batch_size:
            return
        state, action, reward, next_state, done = self._replay_buffer.get_batch()
        qs = self._qnet(state)
        q = qs[np.arange(self._batch_size), action]

        next_qs  = self._qnet_target(next_state)
        next_q = next_qs.max(1)[0]
        next_q.detach_()

        target = (reward + (1 - done) * self._gamma * next_q).to(torch.float)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(q, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()



class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self._buffer = deque(maxlen=buffer_size)
        self._batch_size = batch_size
    
    def __len__(self):
        return len(self._buffer)
    
    def add(self, state: float, action: float, reward: float, next_state: float, done: bool):
        data = (state, action, reward, next_state, done)
        self._buffer.append(data)
    
    def get_batch(self):
        data = random.sample(self._buffer, self._batch_size)
        state = torch.tensor(np.stack([d[0] for d in data])).to(torch.float)
        action = torch.tensor(np.array([d[1] for d in data])).to(torch.long)
        reward = torch.tensor(np.array([d[2] for d in data])).to(torch.float)
        next_state = torch.tensor(np.stack([d[3] for d in data])).to(torch.float)
        done = torch.tensor(np.array([d[4] for d in data])).to(torch.float)
        return state, action, reward, next_state, done

if __name__ == "__main__":
    episodes = 300
    sync_interval = 20
    env = gym.make('CartPole-v0')
    agent = DQNAgent(gamma=0.99, lr=0.001, epsilon=0.1, buffer_size=10000, batch_size=32, action_size=2)
    reward_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_qnet()

        reward_history.append(total_reward)
        if episode % 10 == 0:
            print("episode :{}, total reward : {}".format(episode, total_reward))

    
