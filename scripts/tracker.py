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
from filter import KalmanFilterRADARCamera

class Track():
    unique_id = 0
    def __init__(self):
        # assigning ID
        Track.unique_id += 1
        self.track_id = Track.unique_id
        self.filter = KalmanFilterRADARCamera(vehicle_id=1,
                                            state_dim=4, 
                                            camera_dim=2, 
                                            radar_dim=4, 
                                            control_dim=0, 
                                            dt = 0.1,
                                            first_call_time=0)
        self.state = []
        self.age = 0
        self.miss = 0

    def predict(self):
        self.filter.predict_step(dt)

    def update(self, z_camera = None, z_radar = None, R_radar = None):
        self.filter.update_step(z_camera, z_radar, R_radar)


    # # to be integrated
    # def motion_compensate(self, ego_state, timestamp, track_states):
    #     # ego_state: [x, y, vx, vy, yaw_rate]
    #     dt = timestamp - self.prev_timestamp
    #     vx_dt = ego_state[2] * dt
    #     vy_dt = ego_state[3] * dt
    #     dyaw_dt = ego_state[4] * dt

    #     # todo(heethesh): To invert H or not to?
    #     H = np.asarray([
    #          np.cos(dyaw_dt), np.sin(dyaw_dt), vx_dt,
    #         -np.sin(dyaw_dt), np.cos(dyaw_dt), vy_dt,
    #                        0,               0,     1
    #     ])

    #     # track_state: [x_r, y_r, vx_r, vy_r, x_c, y_c, id]
    #     for i in range(len(track_states)):
    #         # Unpack states
    #         x_r, y_r, vx_r, vy_r, x_c, y_c, track_id = track_states[i]

    #         # Transform the camera and radar poses
    #         radar_vec = np.matmul(H, np.r_[x_r, y_r, 1])[:2]
    #         camera_vec = np.matmul(H, np.r_[x_c, y_c, 1])[:2]

    #         # Pack compensated states
    #         track_states[i] = np.r_[radar_vec, vx_r, vy_r, camera_vec, track_id]

    #     return track_states
    #     # function call
    #     track_states_comp = self.motion_compensate(inputs['ego_state'], inputs['timestamp'], self.get_track_states_with_id())



class Tracker():
    def __init__(self, R_radar, hit_threshold= 1, miss_threshold= 3):
        # tracks is a dictionary with key as an ID and value as a the Track object
        self.R_radar = R_radar
        self.tracks = {}
        self.hit_threshold = hit_threshold
        self.miss_threshold = miss_threshold


    def data_association(self, states_a, states_b, gating_threshold=2):
        ''' Method to solve least cost problem for associating data from two
        input lists of numpy arrays with a cost of 2'''
        # Extract pose from states
        states_a_pose = np.asarray([[state[0], state[1]] for state in states_a])
        states_b_pose = np.asarray([[state[0], state[1]] for state in states_b])

        # Formulate cost matrix
        cost = distance.cdist(states_a_pose, states_b_pose, 'euclidean')
        row_ind, col_ind = linear_sum_assignment(cost)

        # Associate the indices and along with gating
        gate_ind = np.where(cost[row_ind, col_ind] < gating_threshold)
        association = [np.r_[states_a[r], states_b[c]] for r, c in zip(row_ind, col_ind)]
        
        return np.asarray(association)[gate_ind]
    
    def get_track_states_with_id(self):
        '''
        returns an np array with size (n*4+1) with Id as last column 
        '''
        states, track_ids = [], []
        for track_id in self.tracks:
            states.append(self.tracks[track_id].state)
            track_ids.append(track_id)
        return np.c_[states, track_ids]


    def step(self, inputs):

        camera_measurement_used = []
        radar_measurement_used = []

        # below line is redundant
        camera_radar_fused = self.data_association(inputs['camera_measurement'], inputs['radar_measurement'], gating_threshold=2)

        # loop prediction on mesurements and active tracker updating the miss_threshold on each
        for _, track_object in self.tracks.items():
            track_object.predict()
            track_object.motion_compensate(inputs['ego_state'])
        

        active_tracks_with_id = self.get_track_states_with_id()

        # update on camera measurements
        camera_tracker_fused = self.data_association(inputs['camera_measurement'], active_tracks_with_id)
        for track_id, track_object in self.tracks.items():
            for i in range(camera_tracker_fused.shape[0]):
                if camera_tracker_fused[i][-1] == track_id:
                    z_camera = [camera_tracker_fused[i, 0:1]
                    camera_measurement_used.append(camera_tracker_fused[i, 0:2])
                    track_object.miss = 0
                    track_object.age += 1
                    break
                else:
                    z_camera = None
                    track_object.miss += 1
            track_object.update(self.R_radar, z_camera, z_radar= None)

        # update on radar measurements
        radar_tracker_fused = self.data_association(inputs['radar_measurement'], active_tracks_with_id)
        for track_id, track_object in self.tracks.items():
            for i in range(radar_tracker_fused.shape[0]):
                if radar_tracker_fused[i][-1] == track_id:
                    z_radar = [radar_tracker_fused[i: 0:4]]
                    radar_measurement_used.append(radar_tracker_fused[i, 0:4])
                    track_object.miss = 0
                    track_object.age += 1
                    break
                else:
                    z_radar = None
                    track_object.miss += 1
            track_object.update(self.R_radar, z_camera= None, z_radar)

        # for creating tracks with fused camera and radar
        for i in range(camera_radar_fused.shape[0]):
            new_Track_object = None
            if camera_radar_fused[i, 0:2] not in camera_measurement_used:
                if camera_radar_fused[i, 2:6] not in radar_measurement_used:
                    new_Track_object = Track()
                    new_Track_object.update(z_camera = camera_radar_fused[i, 0:2])
                    camera_measurement_used.append(camera_radar_fused[i, 0:2])
                    new_Track_object.update(z_radar = camera_radar_fused[i, 2:6])
                    radar_measurement_used.append(camera_radar_fused[i, 2:6])
                    self.tracks.update({new_Track_object.track_id: new_Track_object})
        
        # for creating tracks with only camera detections
        for i in range(inputs['camera_measurement'].shape[0]):
            new_Track_object = None
            if inputs['camera_measurement'][i] not in camera_measurement_used:
                new_Track_object = Track()
                new_Track_object.update(z_camera= inputs['camera_measurement'][i])
                camera_measurement_used.append(inputs['camera_measurement'][i])

        #for creating tracks with only radar detections
        for i in range(inputs['radar_measurement'].shape[0]):
            new_Track_object = None
            if inputs['radar_measurement'][i] not in radar_measurement_used:
                new_Track_object = Track()
                new_Track_object.update(z_radar= inputs['radar_measurement'][i])
                radar_measurement_used.append(inputs['radar_measurement'][i])

        # Condition to see if all the measurements has been used
        assert radar_measurement_used == inputs['radar_measurement'] , "All the radar measurements has not been used"
        assert camera_measurement_used == inputs['camera_measurement'] , "All the camera measurements has not been used"



if __name__ == '__main__':
    R = np.array([[10,  0,  0,  0],
              [ 0, 10,  0,  0], 
              [ 0,  0, 10,  0],
              [ 0,  0,  0, 10]])
    mot_tracker = Tracker(R)
    
    # Data association test
    states_a = np.asarray([[1,2],[3,4],[5,6]])
    states_b = np.asarray([[1.1,2, 1, 0],[5.1,6.2, 0.5, 0.5],[3.1,4.2, 0, 0],[11,10, 3, 3]])
    print(mot_tracker.data_association(states_a, states_b)) # this returns np array of shape (3, 6)

    # Track ID test
    a = Track()
    print(a.track_id)
    b = Track()
    print(b.track_id)
