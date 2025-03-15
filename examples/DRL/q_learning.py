import numpy as np
import matplotlib.pyplot as plt
from common.gridworld import GridWorld

import dazero.layers as L
import dazero.functions as F
from dazero import Model
from dazero import optimizers


def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(12, 100) # hidden_size
        self.l2 = L.Linear(100, 4)  # action_size

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optimizers.SGD(self.qnet, self.lr)

    def get_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state_vec)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = np.zeros(1)
        else:
            next_qs = self.qnet(next_state)
            next_q = next_qs.max(axis=1)
            next_q._detach()

        target = self.gamma * next_q + reward
        qs = self.qnet(state)
        q = qs[:, action]
        loss = F.mse_loss(target, q)

        self.qnet.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data


if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 2000
    loss_history = []

    for episode in range(episodes):
        state = env.reset()
        state = one_hot(state)
        total_loss, cnt = 0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = one_hot(next_state)

            loss = agent.update(state, action, reward, next_state, done)
            total_loss += loss
            cnt += 1
            state = next_state

        average_loss = total_loss / cnt
        loss_history.append(average_loss)
        if episode % 100 == 0:
            print(f"average_loss: {average_loss: .20f}")

    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.plot(range(len(loss_history)), loss_history)
    plt.show()

    # visualize
    Q = {}
    for state in env.states():
        for action in env.action_space:
            q = agent.qnet(one_hot(state))[:, action]
            Q[state, action] = float(q.data)
    env.render_q(Q)
