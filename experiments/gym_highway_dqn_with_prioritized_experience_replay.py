import math
import matplotlib.pyplot as plt
import random
from collections import deque
import gymnasium as gym
import torch
import numpy as np


class Qnet(torch.nn.Module):
    def __init__(self, action_size: int):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(5, 10)
        self.fc2 = torch.nn.Linear(20, 10)
        self.fc3 = torch.nn.Linear(20, 10)
        self.fc4 = torch.nn.Linear(10, action_size)

    def forward(self, primary_car, others):
        x0 = torch.relu(self.fc1(primary_car))
        x1 = torch.relu(self.fc2(others))
        x = torch.relu(self.fc3(torch.cat((x0, x1), -1)))
        x = self.fc4(x)
        return torch.squeeze(x, 1)


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self._buffer_size = buffer_size
        self._buffer = deque(maxlen=buffer_size)
        self._batch_size = batch_size

    def __len__(self):
        return len(self._buffer)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
        delta: float,
    ):
        self._buffer.append(
            (state, action, reward, next_state, terminated, truncated, delta)
        )

    def get_batch(self):
        delta_sum = sum([d[6] for d in self._buffer])
        weights = [d[6] / delta_sum for d in self._buffer]
        data = random.choices(self._buffer, weights=weights, k=self._batch_size)
        primary_car_state = np.stack([d[0][0] for d in data])
        other_cars_state = np.stack([d[0][1:] for d in data])

        primary_car_state = torch.tensor(primary_car_state).to(torch.float)
        other_cars_state = (
            torch.tensor(other_cars_state)
            .flatten(start_dim=1, end_dim=2)
            .to(torch.float)
        )

        action = torch.tensor(np.stack([d[1] for d in data])).to(torch.long)
        reward = torch.tensor(np.stack([d[2] for d in data])).to(torch.float)
        next_primary_car_state = np.stack([d[3][0] for d in data])
        next_other_cars_state = np.stack([d[3][1:] for d in data])

        next_primary_car_state = torch.tensor(next_primary_car_state).to(torch.float)
        next_other_cars_state = (
            torch.tensor(next_other_cars_state)
            .flatten(start_dim=1, end_dim=2)
            .to(torch.float)
        )
        terminated = torch.tensor(np.stack([d[4] for d in data])).to(torch.float)
        truncated = torch.tensor(np.stack([d[5] for d in data])).to(torch.float)
        return (
            primary_car_state,
            other_cars_state,
            action,
            reward,
            next_primary_car_state,
            next_other_cars_state,
            terminated,
            truncated,
        )


class DQNAgent:
    def __init__(
        self,
        gamma: float,
        lr: float,
        epsilon: float,
        buffer_size: int,
        batch_size: int,
        action_size: int,
    ):
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
            action = np.random.choice(self._action_size)
        else:
            primary_car_state = state[0:1]
            other_cars_state = state[1:]
            primary_car_state = torch.tensor(primary_car_state).to(torch.float)
            other_cars_state = torch.tensor(
                other_cars_state.flatten()[np.newaxis, :]
            ).to(torch.float)
            q = self._qnet(primary_car_state, other_cars_state)
            action = torch.argmax(q).item()
        return action

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ):
        next_primary_car_state = next_state[0:1]
        next_other_cars_state = next_state[1:]
        next_primary_car_state = torch.tensor(next_primary_car_state).to(torch.float)
        next_other_cars_state = torch.tensor(
            next_other_cars_state.flatten()[np.newaxis, :]
        ).to(torch.float)
        q_next = self._qnet_target(next_primary_car_state, next_other_cars_state)
        q_next_max = q_next.max().item()

        primary_car_state = state[0:1]
        other_cars_state = state[1:]
        primary_car_state = torch.tensor(primary_car_state).to(torch.float)
        other_cars_state = torch.tensor(other_cars_state.flatten()[np.newaxis, :]).to(
            torch.float
        )
        q = self._qnet(primary_car_state, other_cars_state).detach().numpy()[0][action]
        delta = abs(reward + self._gamma * (q_next_max - q))

        self._replay_buffer.add(
            state, action, reward, next_state, terminated, truncated, delta
        )

    def update(self):
        (
            primary_car_state,
            other_cars_state,
            actions,
            rewards,
            next_primary_car_state,
            next_other_cars_state,
            terminated,
            _,
        ) = self._replay_buffer.get_batch()
        qs = self._qnet(primary_car_state, other_cars_state)
        q = qs[np.arange(self._batch_size), actions]
        next_qs = self._qnet_target(next_primary_car_state, next_other_cars_state)
        next_q = next_qs.max(1)[0]
        target = (rewards + (1 - terminated) * self._gamma * next_q).to(torch.float)
        loss = torch.nn.functional.mse_loss(q, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.detach().numpy()


if __name__ == "__main__":
    env = gym.make("highway-fast-v0")
    agent = DQNAgent(
        gamma=0.9,
        lr=0.01,
        epsilon=0.5,
        buffer_size=15000,
        batch_size=64,
        action_size=5,
    )
    episodes = 50000

    rewards = []
    total_loss = 0.0
    iter_num = 0
    for episode in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        sum_reward = 0.0
        if episode % 500 == 0 and episode > 0.1:
            agent._epsilon *= 0.9
        while not terminated and not truncated:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.add(state, action, reward, next_state, terminated, truncated)
            if len(agent._replay_buffer) >= agent._replay_buffer._buffer_size:
                loss = agent.update()
            else:
                loss = 0.0
            state = next_state
            sum_reward += reward
            if loss is not None:
                total_loss += loss
                iter_num += 1
        rewards.append(sum_reward)
        print(f"episode: {episode}, sum_reward: {sum_reward}, loss: {loss}")
        if episode % 50 == 0:
            if iter_num != 0:
                print(f"Average loss: {total_loss / iter_num}")
            total_loss = 0.0
            iter_num = 0
            agent.sync_qnet()
            plt.plot(range(len(rewards)), rewards)
            plt.savefig(f"rewards_1.png")
            plt.close()
        if episode == episodes - 1:
            env = gym.make("highway-v0", render_mode="human")
            state, _ = env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
