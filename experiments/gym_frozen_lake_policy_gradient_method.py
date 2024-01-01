import numpy as np
import gym
import torch
import torch.nn.functional as F
from torch.distributions  import Categorical


class Policy(torch.nn.Module):
    def __init__(self, action_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

class Agent:
    def __init__(self, gamma: float, lr: float, epsilon: float, action_size: int):
        self._gamma = gamma
        self._lr = lr
        self._epsilon = epsilon
        self._action_size = action_size
        self._policy = Policy(action_size)
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=lr)
        self._memory = []
    
    def get_action(self, state):
        one_hot_state = F.one_hot(torch.tensor(np.array([state])), num_classes=64).to(torch.float)
        probs = self._policy(one_hot_state)[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs
    
    def add(self, reward, prob):
        self._memory.append((reward, prob))
    
    def update(self, i=0):
        G, loss = 0.0, 0.0
        for reward, _ in reversed(self._memory):
            G = reward + self._gamma * G
        
        for reward, prob in self._memory:
            loss += -torch.log(prob) * G
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._memory = []


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    agent = Agent(gamma=0.9, lr=0.001, epsilon=0.1,  action_size=4)
    episodes = 5000

    for episode in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        sum_reward = 0.0
        while not terminated and not truncated:
            action, probs = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.add(reward, probs[action])
            state = next_state
            sum_reward += reward
        agent.update()
        print(f"episode: {episode}, sum_reward: {sum_reward}")
        if episode == episodes - 1:
            env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human")
            state, _ = env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action, probs = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.add(reward, probs[action])
                state = next_state
