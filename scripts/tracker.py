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
# from filter import KalmanFilterRADARCamera


class Track():
    unique_id = 0
    def __init__(self, state, id):
        self.filter = KalmanFilterRADARCamera(vehicle_id=1,
                                            state_dim=4, 
                                            camera_dim=2, 
                                            radar_dim=4, 
                                            control_dim=0, 
                                            first_call_time=0)
        self.state = state
        self.unique_id = 1
        self.track_id = Track.unique_id
        self.age = 0

    def predict(self):
        self.age += 1
        self.filter.predict_step()
        pass

    def update(self):
        pass

    def motion_compensate(self, ego_state):
        pass


class Tracker():
    def __init__(self, hit_threshold= 1, miss_threshold= 3):
        # tracks is a dictionary with key as an ID and value as a the Track object
        self.tracks = {}
        self.hit_threshold = hit_threshold
        self.miss_threshold = miss_threshold

    def rc_data_association(self, states_a, states_b):
        ''' Method to solve least cost problem for associating data from two 
        input lists of numpy arrays'''
        # Extract relevent states
        states_a_pose = np.asarray(states_a[:, 0:2])
        states_b_pose = np.asarray(states_b[:, 0:2])
        
        # Formulate cost matrix
        cost = distance.cdist(states_a_pose, states_b_pose, 'euclidean')
        row_ind, col_ind = linear_sum_assignment(cost)  
        print (row_ind, col_ind)
        return (np.c_[states_a_pose[row_ind], states_b[col_ind]])
    
    def associate_detections_to_trackers(self, states_a, state_b):
        ''' Method to solve least cost problem for associating data from two 
        input lists of numpy arrays'''

        states_a = []
        for track_id, track_object in self.tracks.items():
            states_a.append(track_object.state, track_id)


        pass

        # # Extract relevent states
        # states_a_pose = np.asarray(states_a[:, 0:4])
        # states_b_pose = np.asarray(states_b[:, 0:4])
        
        # # Formulate cost matrix
        # cost = distance.cdist(states_a_pose, states_b_pose, 'euclidean')
        # row_ind, col_ind = linear_sum_assignment(cost)  
        # print (row_ind, col_ind)
        # return (np.c_[states_a_pose[row_ind], states_b[col_ind]])


    # to be integrated
    def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
        iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

        # Each detecion is mathced (IOU calculated) with each tracks (predicted detections)
        for d,det in enumerate(detections):
            for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det,trk)
        matched_indices = linear_assignment(-iou_matrix) #this tells which detection matches with which tracker - each row contains detecion number and then tracker number

        unmatched_detections = []
        for d,det in enumerate(detections):
            if(d not in matched_indices[:,0]): # first collumn is for detections
            unmatched_detections.append(d)
        unmatched_trackers = []
        for t,trk in enumerate(trackers):
            if(t not in matched_indices[:,1]): # second column is for trackers
            unmatched_trackers.append(t)

        #filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0],m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
            else:
            matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)





    def step(self, inputs):
        fused_states = self.rc_data_association(inputs['camera_tracks'], inputs['radar_tracks'])


        # loop prediction on mesurements and active tracker updating the miss_threshol on each
        for track_id, track_object in self.tracks.items():
            track_object.predict()
            track_object.motion_compensate(inputs['ego_state'])

        # Temporal data associtaion

        


        # Create tracks for each state if they dont already exist
        for state in states:
            self.tracks.appends(Track(state))

        # Call filter with states for each track
        for track in tracks:
            pass




if __name__ == '__main__':
    mot_tracker = Tracker()
    
    # Data association test
    states_a = np.asarray([[1,2],[3,4],[5,6]])
    states_b = np.asarray([[1.1,2, 1, 0],[5.1,6.2, 0.5, 0.5],[3.1,4.2, 0, 0],[11,10, 3, 3]])
    print(mot_tracker.rc_data_association(states_a, states_b))

    # Track ID test
    # a = Track(None)
    # print(a.track_id)
    # b = Track(None)
    # print(b.track_id)
