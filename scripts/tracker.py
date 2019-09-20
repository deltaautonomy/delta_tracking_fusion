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
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# Local python modules
# from filter import KarmeshFillter


class Track():
    unique_id = 0
    def __init__(self, state):
        self.filter = KarmeshFillter()
        self.state = state
        Track.unique_id += 1
        self.track_id = Track.unique_id 

    def predict(self):
        pass

    def update(self):
        pass


class Tracker():
    def __init__(self):
        self.tracks = []

    def data_association(self, states_a, states_b):
        ''' Method to solve least cost problem for associating data from two 
        input lists of numpy arrays'''
        # Extract relevent states
        states_a_pose = np.asarray(states_a[:, 0:2])
        states_b_pose = np.asarray(states_b[:, 0:2])
        
        # Formulate cost matrix
        cost = distance.cdist(states_a_pose, states_b_pose, 'euclidean')
        row_ind, col_ind = linear_sum_assignment(cost)  
        return np.c_[states_a_pose[row_ind], states_b_pose[col_ind]]

    def motion_compensate(self, ego_state):
        pass

    def update(inputs):
        fused_track = []
        new_states = data_association(inputs['camera_tracks'], inputs['radar_tracks'])

        # Do temporal data associtaion

        # Create tracks for each state if they dont already exist
        for state in states:
            self.tracks.appends(Track(state))

        # Call filter with states for each track
        for track in tracks:
            pass

        # Motion compensation


if __name__ == '__main__':
    tracker = Tracker()

    # Data association test
    # states_a = np.asarray([[1,2],[3,4],[5,6]])
    # states_b = np.asarray([[1.1,2, 1, 0],[5.1,6.2, 0.5, 0.5],[3.1,4.2, 0, 0],[11,10, 3, 3]])
    # print(tracker.data_association(states_a, states_b))

    # Track ID test
    # a = Track(None)
    # print(a.track_id)
    # b = Track(None)
    # print(b.track_id)
