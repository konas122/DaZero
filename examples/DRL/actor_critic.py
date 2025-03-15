import gym
import numpy as np

from dazero import Model
from dazero import optimizers
import dazero.layers as L
import dazero.functions as F


class PolicyNet(Model):
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(4, 128)
        self.l2 = L.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.softmax(x)
        return x


class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(4, 128)
        self.l2 = L.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_v = 0.0005
        self.lr_pi = 0.0002
        self.action_size = 2

        self.v = ValueNet()
        self.pi = PolicyNet()
        self.optimizer_v = optimizers.Adam(self.v, self.lr_v)
        self.optimizer_pi = optimizers.Adam(self.pi, self.lr_pi)

    def get_action(self, state):
        state = state[np.newaxis, :]    # add batch axis
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        state = state[np.newaxis, :]    # add batch axis
        next_state = next_state[np.newaxis, :]

        # ========== (1) Update V network ===========
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target._detach()
        v = self.v(state)
        loss_v = F.mse_loss(v, target)

        # ========== (2) Update pi network ===========
        delta = target - v
        delta._detach()
        loss_pi = -F.log(action_prob) * delta

        self.v.zero_grad()
        self.pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()


if __name__ == "__main__":
    episodes = 1000
    env = gym.make('CartPole-v0')
    agent = Agent()
    reward_history = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, prob, reward, next_state, done)

            state = next_state
            total_reward += reward

        reward_history.append(total_reward)
        if episode % 100 == 0:
            print("episode :{}, total reward : {:.1f}".format(episode, total_reward))


    # plot
    from common.utils import plot_total_reward
    plot_total_reward(reward_history)

    # === Play CartPole ===
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        env.render()
    print('Total Reward:', total_reward)
