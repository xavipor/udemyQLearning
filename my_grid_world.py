# -*- coding: utf-8 -*-
"""
Created on Fri May 15 19:36:04 2020

@author: Javier Dominguez

From Udemy 
# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python

"""

from __future__ import print_function, division
from builtins import range
import numpy as np
#(0,0)----> (+j)
#|
#|
#|
#v(+i)

class Grid: 
    def __init__ (self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]
        
    def set(self, rewards, actions):
        # rewards should be a dict of: (i, j): r (row, col): reward
        # actions should be a dict of: (i, j): A (row, col): list of possible actions    
        self.rewards = rewards
        self.actions = actions
        
    def set_state(self,s):
        #Set the agent, using S (state) which is defined as an array 1x2
        self.i =s[0]
        self.j =s[1]
    
    def get_current_state(self):
        return (self.i,self.j)#Current position of the agent
    
    def get_is_terminal(self, s):
        return (s not in self.actions)
    
    def game_over(self):
        return (self.i, self.j) not in self.actions
        #return false, if the current state is not part of the actions dictionary. We have no actions in that state.
        
    def get_all_states(self):
        return set (self.actions.keys()) | set(self.rewards.keys())
        #manera de obtener todos los estados, que son aquellos que aun tienen alguna accion, o que tienen un rewards( para incluir los terminales)
     
    def move(self,action):
        #Check if the action required is between the available ones
        #Remember that action is a dictionary
        if action in self.actions[(self.i,self.j)]: 
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'L':
                self.j -= 1
            elif action == 'R':
                self.j += 1
                
        return self.rewards.get( (self.i,self.j),0 ) #return rewards after moving agent, and 0 if there is no rewards
    
    def undo_move (self, action):
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'L':
            self.j += 1
        elif action == 'R':
                self.j -= 1        
            # range ise an exception if we arrive somewhere we shouldn't be
            # should never happen
        assert(self.current_state() in self.all_states())
    
       
def standard_grid():
  # define a grid that describes the reward for arriving at each state
  # and possible actions at each state
  # the grid looks like this
  # x means you can't go there
  # s means start position
  # number means reward at that state
  # .  .  .  1
  # .  x  . -1
  # s  .  .  .
  g = Grid(3, 4, (2, 0))
  rewards = {(0, 3): 1, (1, 3): -1}
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }
  g.set(rewards, actions)
  return g


def negative_grid(step_cost=-0.1):
  # in this game we want to try to minimize the number of moves
  # so we will penalize every move
  g = standard_grid()
  g.rewards.update({
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
  })
  return g