# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:44:10 2020

@author: Javier Dominguez
"""
from __future__ import print_function, division
from builtins import range
import numpy as np
from my_grid_world import standard_grid, negative_grid
from my_iterative_policy_evaluation import print_values, print_policy


THRESHOLD = 1e-3
GAMMA = 0.9
ALL_ACTIONS = ('U','D','L','R')

# DETERMINISTIC 
# All p(s',r|s,a) = 1 or 0 --> Si quieres ir arriba vas arriba no hay posibilidad de fallo


if __name__ == '__main__':
    
    grid = negative_grid()
    print('rewards')
    print_values(grid.rewards,grid)
    
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_ACTIONS)
        #Assign random actions to each state which have an action. 
        
    # initial policy
    print("initial policy:")
    print_policy(policy, grid)        
    
    V = {}
    states = grid.get_all_states()
    
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            #terminal states are the ones that do not have actions associated
            V[s]=0
    while True:#will break when policy is updated properly. First time its initializes random
    
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]

                if s in policy:
                    a = policy[s]#previosuly it was randomly choosen 
                    grid.set_state(s)
                    r = grid.move(a)
                    V[s] = r + GAMMA* V[grid.get_current_state()]
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))
                    
            if biggest_change < THRESHOLD:
                break
        
        policy_converged= True

        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value= float('-inf')
                for a in ALL_ACTIONS:
                    grid.set_state(s)
                    r= grid.move(a)
                    v = r + GAMMA*V[grid.get_current_state()]
                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[s] = new_a
                if new_a != old_a:
                    policy_converged = False
        if policy_converged:
            break
            
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)                
            
            
    