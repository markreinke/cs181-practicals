import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey
import matplotlib.pyplot as plt
from collections import defaultdict

class QLearner(object):

    def __init__(self, alpha, gamma):
        self.last_state    = None
        self.last_action   = None
        self.last_reward   = None
        self.total         = 0
        self.height = 400
        self.alpha = alpha
        self.gamma = gamma
        self.Q                   = defaultdict(lambda: [0,0])
        self.state_action_counts = defaultdict(lambda: [0,0])
        self.iteration = 1

    def reset(self):
        self.iteration += 1
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    # inspiration for the state representation came from here:
    # https://github.com/georgewu2/cs181-practical4
    def convert_state(self, state):
        binsize = 50
        dist_thresh    = 50
        speed_thresh   = 0

        monkey_tree_top_dist = (state['monkey']['top'] - state['tree']['top']) / binsize
        #monkey_tree_bot_dist = (state['monkey']['bot'] - state['tree']['bot'])/ binsize

        monkey_near_top      = (self.height - state['monkey']['top']) < dist_thresh
        monkey_near_bottom   = state['monkey']['bot']                 < dist_thresh
        monkey_near_tree     = state['tree']['dist']                  < dist_thresh
        monkey_fast          = state['monkey']['vel']                 > speed_thresh
        return (monkey_near_top, monkey_near_bottom, monkey_near_tree, monkey_fast, monkey_tree_top_dist)

    def update_Q(self, state, action):
        last_state_key    = self.convert_state(self.last_state)
        current_state_key = self.convert_state(state)

        max_q = max(self.Q[current_state_key])
        self.Q[last_state_key][action] += self.compute_q(last_state_key, action, max_q)

    def compute_q(self, state_key, action, max_q):
        return (self.alpha / self.state_action_counts[state_key][action]) \
                                                * (self.last_reward \
                                                + self.gamma * max_q \
                                                - self.Q[state_key][action])


    def action_callback(self, state):
        # first time we don't update
        if self.last_state:
            self.update_Q(state, self.last_action)
        state_key = self.convert_state(state)
        # epsilon-greedy
        epsilon = 1.0  / self.iteration
        # take a random action with probability epsilon
        if npr.random() < epsilon:
            new_action = npr.randint(0, 1)
        else:
            max_q  = max(self.Q[state_key])
            new_action =  self.Q[state_key].index(max_q)
        self.state_action_counts[state_key][new_action] += 1

        self.last_action = new_action
        self.last_state  = state

        return self.last_action

    def reward_callback(self, reward):
        self.last_reward = reward
        self.total = self.total + reward

def run_games(learner, hist, iters = 50, t_len = 1):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        # Save score history.
        hist.append(swing.score)
        # Reset the state of the learner.
        learner.reset()

    return

if __name__ == '__main__':

    # Select agent.
    agent = QLearner(0.25, 0.75)

    # Empty list to save history.
    hist = []

    # Run games.
    run_games(agent, hist)

    # Save history.
    np.save('hist', np.array(hist))
