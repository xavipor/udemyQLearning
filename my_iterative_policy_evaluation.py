# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:17:15 2020

@author: Javier Dominguez
"""

from __future__ import print_function, division
from builtins import range
import numpy as np
from my_grid_world import standard_grid

#Given a Policy, its value fuction will be calculated. p(a|s) will be modeled as 
#a uniform distribution. Each action will have the same prob. p(s',r| a,s) this is 
#not modelled since there will be no randomness in the chance of doing an action. If u want
#to move up, you will, there is no chances of movin down for any case. 

THRESHOLD = 1e-3
GAMMA = 1

def print_values (V,g):
    for i in range(g.rows):
        print("--------------------------")
        for j in range(g.cols):
            v = V.get((i,j),0)
            if v>=0:#por motivos esteticos, para que cuadre bien.
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")
        print("")
        
    
def print_policy(P,g):
    for i in range(g.rows):
        print("--------------------------")
        for j in range(g.cols):
            a = P.get((i,j), ' ')
            print("  %s  |" % a, end="")
        print("")
        
if __name__ == "__main__":
    
    grid = standard_grid()
    states = grid.get_all_states()
    V={}
    for s in states:
        V[s]= 0 #Go through all states and initialize the value to 0
    print("values for uniformly random actions First Time:")
    print_values(V, grid)
    print("\n\n")      
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]
            
            if s in grid.actions: #let out terminal states
                new_v = 0
                p_a = 1.0/len(grid.actions[s])
                
                for a in grid.actions[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p_a * (r + GAMMA * V[grid.get_current_state()])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        
        print_values(V, grid)        
        if biggest_change < THRESHOLD:  
            break              
    print("values for uniformly random actions:")
    print_values(V, grid)
    print("\n\n")        
    
################## Lets see the effect of use fixed policy ###
    policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
    }    
    print_policy(policy, grid)
    GAMMA = 0.9
    V={}
    for s in states:
        V[s]= 0
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]
            
            if s in policy: #Diferent now, we dont want to iterate over all actions in state, since 
            #we have a fixed policy
                a = policy[s]
                grid.set_state(s)
                
                r = grid.move(a)
                V[s] = r + GAMMA*V[grid.get_current_state()]
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        if biggest_change < THRESHOLD:  
            break 
    print("values for fixed policy:")
    print_values(V, grid)        
    #cuanto mas lejos del objetivo, el valor tiene el decay de 0.9. Así cada paso atrás es un 
    #multiplo de 0.9
                
        
    