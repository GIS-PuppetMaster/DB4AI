import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from dqn_env import DQN_env
import time
Episodes = 1000

class dqn_agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999995
        self.learning_rate = 0.0001
        self.loss=[]
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def reply(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.history=self.model.fit(state, target_f, epochs=1, verbose=0)
            b = abs(float(self.history.history['loss'][0]))
            self.loss.append(b)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
    def getloss(self):
        return self.loss


def dqn_train(dqn_time_limit):
    env = DQN_env()
    state_size = env.nStates
    action_size = env.nActions
    agent = dqn_agent(state_size, action_size)
    done = False
    batch_size = 32
    a = time.time()
    action_list1=[]
    for e in range(Episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        action_list = []
        for times in range(100):
            action = agent.act(state)
            if action not in action_list:
                action_list.append(action)
            next_state, reward, done = env.step(action, state)
            reward = reward if not done else -1
            agent.memorize(state, action, reward, next_state, done)

            state = next_state
            if done:
                print("action list: ")
                print(action_list)
                action_list1 = action_list
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, Episodes, times, agent.epsilon))
                with open("features.txt", "a+") as file:
                    for ac in action_list:
                        file.write(str(ac) + ' ')
                    file.write('\n')
                break
            if len(agent.memory) > batch_size:
                agent.reply(batch_size)
            b = time.time()
            if b - a > dqn_time_limit:
                with open('Selected_features.txt', 'w') as f:
                    for i in range(len(action_list1)):
                        f.write(str(action_list1[i]) + '\t')
                return action_list1

    with open('Selected_features.txt', 'w') as f:
        f.write(str(len(action_list1))+'\t')
        for i in range(len(action_list1)):
            f.write(str(action_list1[i]) + '\t')
    return action_list1




