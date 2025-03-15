import gym
import numpy as np

from dazero import Model
from dazero import optimizers
import dazero.layers as L
import dazero.functions as F


class Policy(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(4, 128)
        self.l2 = L.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.pi, self.lr)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        self.pi.zero_grad()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G

        for reward, prob in self.memory:
            loss += -F.log(prob) * G

        loss.backward()
        self.optimizer.step()
        self.memory = []


if __name__ == "__main__":
    episodes = 5000
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

            agent.add(reward, prob)
            state = next_state
            total_reward += reward

        agent.update()

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
