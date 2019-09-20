#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Apoorv Singh
Email   : apoorvs@andrew.cmu.edu
Version : 1.0.0
Date    : Sep 17, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Handle paths and OpenCV import
from init_paths import *

# Built-in modules

# External modules
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

class Tracker():
    def __init__(self):
        pass

    def data_association(self, states_a, states_b):
        ''' Method to solve least cost problem for associating data from two 
        input lists of numpy arrays'''

        # Extract relevent states
        states_a_pose = np.asarray(states_a[:,0:2])
        states_b_pose = np.asarray(states_b[:,0:2] )
        # Formulate cost matrix
        cost = distance.cdist(states_a_pose, states_b_pose, 'euclidean')
        row_ind, col_ind = linear_sum_assignment(cost)  

        return np.c_[states_a_pose[row_ind], states_b_pose[col_ind]]

    def update(inputs):
        oncoming_states = data_association(inputs['camera_tracks'], inputs['radar_tracks'])


if __name__ == '__main__':
    tracker = Tracker()
    states_a = np.asarray([[1,2],[3,4],[5,6]])
    states_b = np.asarray([[1.1,2],[5.1,6.2],[3.1,4.2],[11,10]])
    print(tracker.data_association(states_a, states_b))

