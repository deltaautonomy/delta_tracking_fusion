#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Apoorv Singh, Heethesh Vhavle
Email   : apoorvs@andrew.cmu.edu
Version : 1.0.1
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


class Tracklet():
    unique_id = 0
    def __init__(self, timestamp, z_radar=None, z_camera=None):
        # Assign new unique track ID
        Tracklet.unique_id += 1
        self.track_id = Tracklet.unique_id

        # Tracklet streaks
        self.age = 0
        self.hits = 0
        self.misses = 0

        # Sensor fusion
        self.state = None
        self.last_update_time = timestamp
        self.filter = KalmanFilterRADARCamera(
            vehicle_id = self.track_id,
            state_dim = 4,
            camera_dim = 2,
            radar_dim = 4,
            control_dim = 0,
            dt = 0.1,
            first_call_time = timestamp
        )

        self.update(timestamp, z_radar=z_radar, z_camera=z_camera)

    def predict(self, timestamp):
        self.age += 1
        self.state = self.filter.predict_step(timestamp)

    def update(self, timestamp, z_radar=None, z_camera=None):
        assert not(z_radar is None and z_camera is None), \
            'Atleast one measurement is required to update the filter'

        # Update filters with current measurements
        if timestamp > self.last_update_time: self.hits += 1
        self.state = self.filter.update_step(z_radar=z_radar, z_camera=z_camera)
        self.last_update_time = timestamp


class Tracker():
    def __init__(self, hit_window=3, miss_window=3):
        self.tracks = {}
        self.hit_window = hit_window
        self.miss_window = miss_window
        self.radar_noise = np.eye(4) * 100
        self.prev_ego_state = None
        self.prev_timestamp = None

    def data_association(self, states_a, states_b, gating_threshold=2):
        ''' Method to solve least cost problem for associating data from two 
        input lists of numpy arrays'''
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
        '''Returns a np.array with size (n, 4 + 1) with ID in the last column'''
        states, track_ids = [], []
        for track_id in self.tracks:
            states.append(self.tracks[track_id].state)
            track_ids.append(track_id)
        return np.c_[states, track_ids]

    def motion_compensate(self, ego_state, timestamp, track_states):
        # ego_state: [x, y, vx, vy, yaw_rate]
        dt = timestamp - self.prev_timestamp
        vx_dt = ego_state[2] * dt
        vy_dt = ego_state[3] * dt
        dyaw_dt = ego_state[4] * dt

        # todo(heethesh): To invert H or not to?
        H = np.asarray([
             np.cos(dyaw_dt), np.sin(dyaw_dt), vx_dt,
            -np.sin(dyaw_dt), np.cos(dyaw_dt), vy_dt,
                           0,               0,     1
        ])

        # Transform the state poses
        # track_state: [x, y, vx, vy, id]
        for i in range(len(track_states)):
            x, y, vx, vy, track_id = track_states[i]
            state_comp = np.matmul(H, np.r_[x, y, 1])[:2]
            track_states[i] = np.c_[state, track_id]

        return track_states

    def update(self, inputs):
        #------------------- PREDICT / MOTION COMPENSATE -------------------#

        # Predict new states for all tracklets for the next timestep
        for track_id in self.tracks:
            self.tracks[track_id].predict(timestamp)

        # Transform all tracklet states by ego motion
        track_states_comp = self.motion_compensate(inputs['ego_state'],
            inputs['timestamp'], self.get_track_states_with_id())

        #----------------------- UPDATE OLD TRACKLETS -----------------------#

        # Assign temporary ID for each detection to keep track of its association status
        radar_dets = np.c_[inputs['radar_measurement'], np.arange(len(inputs['radar_measurement']))]
        camera_dets = np.c_[inputs['camera_measurement'], np.arange(len(inputs['camera_measurement']))]

        # Keep the status of which tracklets are being updated and not
        track_updated = {track_id: False for track_id in self.tracks.keys()}
        unmatched_radar_dets, unmatched_camera_dets = radar_dets, camera_dets

        # Temporal data association using RADAR detections with compensated states
        if len(track_states_comp) and len(radar_dets):
            matched_radar_dets = self.data_association(radar_dets, track_states_comp)
            unmatched_radar_dets = radar_dets[np.setdiff1d(radar_dets[:, -1],
                matched_radar_dets[:, 4]).astype('int')]

            # Update tracklets with RADAR measurements
            for track_id in matched_radar_dets[:, -1]:
                self.tracks[track_id].update(inputs['timestamp'], z_radar=matched_radar_dets[:4])
                track_updated[track_id] = True

        # Temporal data association using camera detections with compensated states
        if len(track_states_comp) and len(camera_dets):
            matched_camera_dets = self.data_association(camera_dets, track_states_comp)
            unmatched_camera_dets = camera_dets[np.setdiff1d(camera_dets[:, -1],
                matched_camera_dets[:, 2]).astype('int')]

            # Update tracklets with camera measurements
            for track_id in matched_camera_dets[:, -1]:
                self.tracks[track_id].update(inputs['timestamp'], z_camera=matched_camera_dets[:2])
                track_updated[track_id] = True

        # Updated miss count for all those tracklets that did not get updated
        for track_id in track_updated:
            if not track_updated[track_id]:
                self.tracks[track_id].misses += 1

        #----------------------- CREATE NEW TRACKLETS -----------------------#

        # Now first try to associate the unmatched camera/RADAR data together
        # to avoid creating new tracklets if they are the same target to be tracked
        if len(unmatched_radar_dets) and len(unmatched_camera_dets):
            matched_camera_radar_dets = self.data_association(unmatched_radar_dets,
                unmatched_camera_dets)

            # Create new tracklets if we got matches
            for meas in matched_camera_radar_dets:
                new_track = Tracklet(inputs['timestamp'], z_radar=meas[:4], z_camera=meas[5:7])
                self.tracks[track.track_id] = new_track

            # Update unmatched lists
            unmatched_radar_dets = unmatched_radar_dets[np.setdiff1d(radar_dets[:, -1],
                matched_camera_radar_dets[:, 4]).astype('int')]
            unmatched_camera_dets = camera_dets[np.setdiff1d(camera_dets[:, -1],
                matched_camera_radar_dets[:, 7]).astype('int')]

        # Creating tracklets with only unmatched RADAR detections
        for meas in unmatched_radar_dets:
            new_track = Tracklet(inputs['timestamp'], z_radar=meas[:4])
            self.tracks[track.track_id] = new_track

        # Creating tracklets with only unmatched camera detections
        for meas in unmatched_camera_dets:
            new_track = Tracklet(inputs['timestamp'], z_camera=meas[:2])
            self.tracks[track.track_id] = new_track

        #--------------------- HANDLE TRACKLET STREAKS ---------------------#

        # Purge old tracklets with no updates
        for track_id in self.tracks:
            if self.tracks[track_id].misses >= self.miss_window:
                del self.tracks[track_id]            

        # Return confident tracklets
        fused_tracks = []
        for track_id in self.tracks:
            if self.tracks[track_id].hits >= self.hit_window:
                fused_tracks.append(np.c_[self.tracks[track_id].state.copy(), track_id])

        # Store data for the next timestep 
        self.prev_ego_state = inputs['ego_state']
        self.prev_timestamp = inputs['timestamp']

        return np.asarray(fused_tracks)


if __name__ == '__main__':
    # R = np.array([[10,  0,  0,  0],
    #           [ 0, 10,  0,  0], 
    #           [ 0,  0, 10,  0],
    #           [ 0,  0,  0, 10]])
    # mot_tracker = Tracker(R)
    
    tracker = Tracker()

    # Data association test
    # states_a = np.asarray([[1,2],[3,4],[0, 0],[5,6]])
    # states_b = np.asarray([[1.1,2, 1, 0],[5.1,6.2, 0.5, 0.5],[3.1,4.2, 0, 0],[11,10, 3, 0]])
    # print(tracker.data_association(states_a, states_b))

    # Track ID test
    # a = Track(None)
    # print(a.track_id)
    # b = Track(None)
    # print(b.track_id)

    # Tracker update test
    inputs = {}
    inputs['timestamp'] = 0
    inputs['camera_measurement'] = np.asarray([[1, 2], [3, 4], [0, 0], [5, 6]])
    inputs['radar_measurement'] = np.asarray([[1.1, 2, 1, 0], [5.1, 6.2, 0.5, 0.5], [3.1, 4.2, 0, 0], [11, 10, 3, 0]])
    tracker.update(inputs)
