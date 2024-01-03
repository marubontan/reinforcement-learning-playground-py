import random
import numpy as np
from collections import deque
import gym
import torch
import torch.nn.functional as F


class DuelingQnet(torch.nn.Module):
    def __init__(self, action_size: int):
        super(DuelingQnet, self).__init__()
        self.fc1 = torch.nn.Linear(64, 128)

        self.value_stream = torch.nn.Linear(128, 1)
        self.advantage_stream = torch.nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self._buffer = deque(maxlen=buffer_size)
        self._batch_size = batch_size

    def __len__(self):
        return len(self._buffer)

    def add(
        self,
        state: float,
        action: float,
        reward: float,
        next_state: float,
        terminated: bool,
        truncated,
    ):
        self._buffer.append((state, action, reward, next_state, terminated, truncated))

    def get_batch(self):
        data = random.sample(self._buffer, self._batch_size)
        state = F.one_hot(
            torch.tensor(np.stack([d[0] for d in data])), num_classes=64
        ).to(torch.float)
        action = torch.tensor(np.stack([d[1] for d in data])).to(torch.long)
        reward = torch.tensor(np.stack([d[2] for d in data])).to(torch.float)
        next_state = F.one_hot(
            torch.tensor(np.stack([d[3] for d in data])), num_classes=64
        ).to(torch.float)
        terminated = torch.tensor(np.stack([d[4] for d in data])).to(torch.float)
        truncated = torch.tensor(np.stack([d[5] for d in data])).to(torch.float)
        return state, action, reward, next_state, terminated, truncated


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
        self._qnet = DuelingQnet(action_size)
        self._qnet_target = DuelingQnet(action_size)
        self._optimizer = torch.optim.Adam(self._qnet.parameters(), lr=lr)

    def sync_qnet(self):
        self._qnet_target.load_state_dict(self._qnet.state_dict())

    def get_action(self, state):
        if np.random.rand() < self._epsilon:
            return np.random.choice(self._action_size)
        else:
            state = F.one_hot(torch.tensor(np.array([state])), num_classes=64)[
                np.newaxis, :
            ].to(torch.float)
            qs = self._qnet(state)
            action = qs.argmax().item()
        return action

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
    ):
        self._replay_buffer.add(
            state, action, reward, next_state, terminated, truncated
        )
        if len(self._replay_buffer) < self._replay_buffer._batch_size:
            return
        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
        ) = self._replay_buffer.get_batch()
        qs = self._qnet(states)
        q = qs[np.arange(self._batch_size), actions]

        next_qs = self._qnet_target(next_states)
        next_q = next_qs.max(1)[0]
        next_q.detach_()

        target = (rewards + (1 - terminated) * self._gamma * next_q).to(torch.float)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(q, target)
        self._optimizer.zero_grad()
        loss.backward()

        self._optimizer.step()
        return loss


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    agent = DQNAgent(0.9, 0.001, 0.1, 1000, 64, 4)
    reward_history = []
    episodes_per_iteration = 500
    evaluation_episodes = 30
    episodes = 5000

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = agent.get_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)

            loss = agent.update(state, action, reward, new_state, terminated, truncated)
            state = new_state
            total_reward += reward

        reward_history.append(total_reward)
        if episode % episodes_per_iteration == 0:
            eval_total_reward = 0.0
            for eval_episode in range(evaluation_episodes):
                eval_state, _ = env.reset()
                eval_terminated = False
                eval_truncated = False
                while True:
                    eval_action = agent.get_action(eval_state)
                    (
                        eval_new_state,
                        eval_reward,
                        eval_terminated,
                        eval_truncated,
                        _,
                    ) = env.step(eval_action)
                    if eval_terminated or eval_truncated:
                        break
                    eval_state = eval_new_state
                    eval_total_reward += eval_reward
                eval_total_reward += total_reward
            print(
                f"episode :{episode}, total reward : {eval_total_reward}, avg reward: {eval_total_reward / evaluation_episodes}"
            )

        if episode % 50 == 0:
            agent.sync_qnet()

        if episode == episodes - 1:
            env = gym.make(
                "FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human"
            )
            eval_state, _ = env.reset()
            eval_terminated = False
            eval_truncated = False
            while True:
                eval_action = agent.get_action(eval_state)
                (
                    eval_new_state,
                    eval_reward,
                    eval_terminated,
                    eval_truncated,
                    _,
                ) = env.step(eval_action)
                if eval_terminated or eval_truncated:
                    break
                eval_state = eval_new_state
            env.render()
