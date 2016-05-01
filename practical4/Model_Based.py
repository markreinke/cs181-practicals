# Imports.
import csv
import numpy as np
from numpy import random
import sys
from SwingyMonkey import SwingyMonkey
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict

# Variables
x_grid_px = 20
y_grid_px = 20

class ModelBasedLearner:

    def __init__(self, gamma):
        self.previous_state  = None
        self.previous_action = None
        self.previous_reward = None
        self.screen_height = 400
        self.screen_width = 600
        
        self.gamma = gamma
        
        # Get total number of states given the grid 
        self.total_states = ((self.screen_width/x_grid_px)*2 + 1) * ((self.screen_height/y_grid_px)*2 + 1)
        
        # Keep track of all states while running
        self.list_of_states = [] #To store list of all States
        
        ### Initialisation of Q function, Rewards, Transition
        self.Q                   = defaultdict(lambda: [0,0]) #Q[States][Action]
        self.R                   = defaultdict(lambda: [[0,1] , [0,1]]) # R[States][action][n,tot]
        self.T                   = defaultdict(lambda: [[0,1] , [0,1]]) # T[States_prev, states][action][n,tot]
        

    def convert_state(self, state):
        binsize = 50
        dist_thresh    = 50
        speed_thresh   = 0
        
        monkey_tree_top_dist = (state['monkey']['top'] - state['tree']['top']) / binsize   

        monkey_near_top      = (self.screen_height - state['monkey']['top']) < dist_thresh
        monkey_near_bottom   = state['monkey']['bot']                 < dist_thresh
        monkey_near_tree     = state['tree']['dist']                  < dist_thresh
        monkey_fast          = state['monkey']['vel']                 > speed_thresh
        return (monkey_near_top, monkey_near_bottom, monkey_near_tree, monkey_fast, monkey_tree_top_dist)


    def reset(self):
        self.previous_state  = None
        self.previous_action = None
        self.previous_reward = None

    # computes optimal policy and value function value at a given state
    def get_best_value(self, state):   
        if self.Q[state][1] > self.Q[state][0]:
            return (self.Q[state][1],1)
        else:
            return (self.Q[state][0],0)
    
    def UpdateT(self, state_i):
        self.T[self.previous_state, state_i][self.previous_action][0] += 1
        # Update Count
        for state in self.list_of_states:
            self.T[self.previous_state, state][self.previous_action][1] += 1
                
    def UpdateR(self):
        self.R[self.previous_state][self.previous_action][0] += self.previous_reward
        # Update Count
        self.R[self.previous_state][self.previous_action][1] += 1
        
    def GetProbability(self, previous_state, state, action):
        # Prob = N / ToT
        return self.T[previous_state,state][action][0] / self.T[previous_state,state][action][1]
    
    def GetAverageR(self, state, action):
        return self.R[state][action][0] / self.R[state][action][1]
        
    def UpdateQ(self, state_i):
        
        # Update Q function (action = 0)
        prob_x_Expectation = 0.0
        for state in self.list_of_states:
            prob_x_Expectation += (self.GetProbability(state_i, state, 0) * self.get_best_value(state)[0])
        self.Q[state_i][0] = self.GetAverageR(state_i, 0) + (self.gamma * prob_x_Expectation)

        # Update Q function (action = 1)
        prob_x_Expectation = 0.0
        for state in self.list_of_states:
            prob_x_Expectation += (self.GetProbability(state_i,state,1) * self.get_best_value(state)[0])
        self.Q[state_i][1] = self.GetAverageR(state_i, 1) + (self.gamma * prob_x_Expectation)
             
    
    def action_callback(self, state):
        state_i = self.convert_state(state)

        if self.previous_state != None:
            self.UpdateT(state_i)
            self.UpdateR()
            self.UpdateQ(state_i)

            # Decide which action
            self.previous_action = np.argmax([self.Q[state_i][0],self.Q[state_i][1]])
                
        else:
            self.previous_action = random.random() < 0

        #Memory Save Previous state
        self.previous_state = state_i

        return self.previous_action

    def reward_callback(self, reward):
        self.previous_reward = reward

def run_games(learner, hist, iters = 100, t_len = 100):
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
	agent = ModelBasedLearner(0.75)

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 100, 10)

	# Save history. 
	np.save('hist',np.array(hist))


