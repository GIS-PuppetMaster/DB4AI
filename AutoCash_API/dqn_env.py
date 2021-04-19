import numpy as np
import Features

feature_list,feature_dict=Features.featurelist_choose()
class DQN_env:
    def __init__(self):
        self.actionSpace = feature_list
        self.nActions = len(self.actionSpace)
        self.nStates = 2 ** self.nActions
        self.startState = 0
        self.x_threshold=1
        self.state = self.startState
        self.actionlist = []

    def _buildEnv(self):
        origin = np.array([20, 20])

    def step(self, action, state):
        #next_s = s | (1 << action)
        tmp = []
        for i in state:
            tmp.append(i | (1 << action))
        next_s = np.array(tmp)
        action=feature_dict[action]
        if action == 0:
            reward = 0.95445
            # done = True
        elif action == 1:
            reward = 0.940983
            # done = True
        elif action == 2:
            reward = 0.942667
            # done = True
        elif action == 3:
            reward = 0.922467
            # done = True
        elif action == 4:
            reward = 0.8989
            # done = True
        elif action == 5:
            reward = 0.834933
            # done = True
        elif action == 6:
            reward = 0.81305
            # done = True
        elif action == 7:
            reward = 0.860183
            # done = True
        elif action == 8:
            reward = 0.9393
            # done = True
        elif action == 9:
            reward = 0.7575
            # done = True
        elif action == 10:
            reward = 0.8484
            # done = True
        elif action == 11:
            reward = 0.804633
            # done = True
        elif action == 12:
            reward = 0.865233
            # done = True
        elif action == 13:
            reward = 0.7171
            # done = True
        elif action == 14:
            reward = 0.9393
            # done = True
        elif action == 15:
            reward = 0.855133
            # done = True
        elif action == 16:
            reward = 0.8484
            # done = True
        elif action == 17:
            reward = 0.725517
            # done = True
        elif action == 18:
            reward = 0.7878
            # done = True
        elif action == 19:
            reward = 0.895533
            # done = True
        elif action == 20:
            reward = 0.846717
            # done = True
        elif action == 21:
            reward = 0.779383
            # done = True
        elif action == 22:
            reward = 0.850083
            # done = True
        #if action not in self.actionlist:
        self.actionlist.append(action)
        if len(self.actionlist) > 8:
            done = True
            self.actionlist = []
        else:
            done = False
        return next_s, reward, done

    def render(self):
        self.update()

    def reset(self):
        l = []
        for i in range(2 ** self.nActions):
            l.append(i)
        return np.array(l)


