#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle, Apoorv Singh
Email   : heethesh@cmu.edu, apoorvs@andrew.cmu.edu
Version : 1.0.1
Date    : Sep 17, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Handle paths and OpenCV import
from init_paths import *

# Built-in modules
import pdb

# ROS modules
import rospy

# External modules
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# Local python modules
from filter import KalmanFilterRADARCamera


class Tracklet():
    unique_id = 0
    def __init__(self, timestamp, z_radar=None, z_camera=None, verbose=False):
        # Assign new unique track ID
        Tracklet.unique_id += 1
        self.track_id = Tracklet.unique_id
        self.verbose = verbose

        # Tracklet streaks
        self.age = 0
        self.hits = 0
        self.misses = 0

        # RADAR measurement noise
        self.std_radar_x = 0.5  # 3
        self.std_radar_y = 0.5  # 1
        self.std_radar_vx = 0.5  # 1.5
        self.std_radar_vy = 0.5  # 1
        self.R_radar = np.diag([self.std_radar_x ** 2, self.std_radar_y ** 2,
                                self.std_radar_vx ** 2, self.std_radar_vy ** 2])

        # Camera measurement noise
        self.std_camera_x = 10000
        self.std_camera_y = 10000
        self.R_camera = np.diag([self.std_camera_x ** 2, self.std_camera_y ** 2])

        # Sensor fusion
        self.state = None
        self.state_cov = None
        self.last_update_time = timestamp
        self.filter = KalmanFilterRADARCamera(vehicle_id=self.track_id,
                                              state_dim=4,
                                              camera_dim=2,
                                              radar_dim=4,
                                              control_dim=0,
                                              dt=0.01,
                                              first_call_time=timestamp,
                                              verbose=False)

        if self.verbose: print('New track:', z_camera, z_radar)
        self.update(timestamp, z_radar=z_radar, z_camera=z_camera)

    def predict(self, timestamp):
        self.age += 1
        self.state = self.filter.predict_step(timestamp)
        self.state_cov = self.filter.P
        if self.verbose: print ('Predicted state from kalman filter at ', timestamp, ' is ', self.state.flatten())

    def update(self, timestamp, z_radar=None, z_camera=None):
        assert not(z_radar is None and z_camera is None), \
            'Atleast one measurement is required to update the filter'

        # Update filters with current measurements
        if timestamp > self.last_update_time: self.hits += 1
        self.state = self.filter.update_step(z_radar=z_radar, z_camera=z_camera,
                                             R_radar=self.R_radar, R_camera=self.R_camera)
        self.state_cov = self.filter.P
        self.last_update_time = timestamp
        if self.verbose: print('Updated state from kalman filter', self.state.flatten())
            

class Tracker():
    def __init__(self, hit_window=1, miss_window=5, verbose=False):
        self.tracks = {}
        self.hit_window = hit_window
        self.miss_window = miss_window
        self.prev_ego_state = None
        self.prev_timestamp = None
        self.verbose = verbose

    def data_association(self, states_a, states_b, gating_threshold=10):
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
            states.append(self.tracks[track_id].state.flatten())
            track_ids.append(track_id)
        return np.c_[states, track_ids]

    def motion_compensate(self, ego_state, timestamp, track_states, inverse=True):
        # Motion compensation is not used, radar detection are
        # instead velocity compensated later
        return track_states

        # ego_state: [x, y, vx, vy, yaw_rate]
        dt = (timestamp - self.prev_timestamp)
        vx_dt = ego_state[2] * dt
        vy_dt = ego_state[3] * dt
        dyaw_dt = ego_state[4] * dt

        H = np.asarray([[np.cos(dyaw_dt), -np.sin(dyaw_dt), vx_dt],
                        [np.sin(dyaw_dt),  np.cos(dyaw_dt), vy_dt],
                        [              0,                0,     1]])

        if inverse: H = np.linalg.inv(H)

        # Transform the state poses
        # track_state: [x, y, vx, vy, id]
        for i in range(len(track_states)):
            x, y, vx, vy, track_id = track_states[i]
            pose = np.matmul(H, np.r_[x, y, 1])[:2]
            track_states[i] = np.r_[pose, vx, vy, track_id]

        return track_states

    def velocity_compensation(self, ego_state, radar_msgs):
        vx = ego_state[2]
        vy = ego_state[3]

        for i in range(len(radar_msgs)):
            radar_msgs[i, 2] -= vx
            radar_msgs[i, 3] -= vy
        return radar_msgs

    def update(self, inputs):
        if self.verbose: print('\n\n---------------- NEW FRAME ----------------\n')

        #------------------- PREDICT / MOTION COMPENSATE -------------------#

        # Predict new states for all tracklets for the next timestep
        for track_id in self.tracks:
            self.tracks[track_id].predict(inputs['timestamp'])

        # Transform all tracklet states by ego motion
        track_states_comp = []
        if self.prev_timestamp is not None:
            track_states_comp = self.motion_compensate(inputs['ego_state'],
                inputs['timestamp'], self.get_track_states_with_id())

        # Assign temporary ID for each detection to keep track of its association status
        radar_dets = np.c_[inputs['radar'], np.arange(len(inputs['radar']))]
        camera_dets = np.c_[inputs['camera'], np.arange(len(inputs['camera']))]

        # Compensating velocity of the input radar measurements with the ego vehicle velocity
        radar_dets = self.velocity_compensation(inputs['ego_state'], radar_dets)
        if self.verbose: print('Raw radar detections:\n', radar_dets)

        #----------------------- UPDATE OLD TRACKLETS -----------------------#

        # Keep the status of which measurements are being used and not
        radar_matched_ids, camera_matched_ids = set(), set()
        radar_unmatched_ids = set(range(len(radar_dets)))
        camera_unmatched_ids = set(range(len(camera_dets)))

        # Keep the status of which tracklets are being updated and not
        track_updated = {track_id: False for track_id in self.tracks.keys()}
        if len(track_states_comp) and len(radar_dets):
            # Temporal data association using RADAR detections with compensated states
            matched_radar_dets = self.data_association(radar_dets, track_states_comp)

            # Update tracklets with RADAR measurements
            for meas in matched_radar_dets:
                track_id = meas[-1]
                self.tracks[track_id].update(inputs['timestamp'], z_radar=meas[:4])
                track_updated[track_id] = True

                # Update matched measurements ID set
                radar_matched_ids.add(int(meas[4]))

            # Update unmatched measurements ID set
            radar_unmatched_ids -= radar_matched_ids

        if len(track_states_comp) and len(camera_dets):
            # Temporal data association using camera detections with compensated states
            matched_camera_dets = self.data_association(camera_dets, track_states_comp)

            # Update tracklets with camera measurements
            for meas in matched_camera_dets:
                track_id = meas[-1]
                self.tracks[track_id].update(inputs['timestamp'], z_camera=meas[:2])
                track_updated[track_id] = True

                # Update matched measurements ID set
                camera_matched_ids.add(int(meas[2]))

            # Update unmatched measurements ID set
            camera_unmatched_ids -= camera_matched_ids

        # Update miss count for all those tracklets that did not get updated
        for track_id in track_updated:
            if not track_updated[track_id]:
                self.tracks[track_id].misses += 1
            else:
                self.tracks[track_id].misses = 0

        #----------------------- CREATE NEW TRACKLETS -----------------------#

        # Now first try to associate the unmatched camera/RADAR data together
        # to avoid creating new tracklets if they are the same target to be tracked
        unmatched_radar_dets = radar_dets[list(radar_unmatched_ids)]
        unmatched_camera_dets = camera_dets[list(camera_unmatched_ids)]
        if len(unmatched_radar_dets) and len(unmatched_camera_dets):
            matched_camera_radar_dets = self.data_association(unmatched_radar_dets,
                unmatched_camera_dets)

            # Create new tracklets if we got matches
            for meas in matched_camera_radar_dets:
                new_track = Tracklet(inputs['timestamp'], z_radar=meas[:4],
                    z_camera=meas[5:7], verbose=self.verbose)
                self.tracks[new_track.track_id] = new_track

                # Updated matched measurements ID set
                radar_matched_ids.add(int(meas[4]))
                camera_matched_ids.add(int(meas[7]))

            # Update unmatched measurements ID set
            radar_unmatched_ids -= radar_matched_ids
            camera_unmatched_ids -= camera_matched_ids

        # Creating tracklets with only unmatched RADAR detections
        unmatched_radar_dets = radar_dets[list(radar_unmatched_ids)]
        for meas in unmatched_radar_dets:
            new_track = Tracklet(inputs['timestamp'],
                z_radar=meas[:4], verbose=self.verbose)
            self.tracks[new_track.track_id] = new_track

            # Updated matched measurements ID set
            radar_matched_ids.add(int(meas[4]))

        # Update unmatched measurements ID set
        radar_unmatched_ids -= radar_matched_ids

        # Creating tracklets with only unmatched camera detections
        unmatched_camera_dets = camera_dets[list(camera_unmatched_ids)]
        for meas in unmatched_camera_dets:
            new_track = Tracklet(inputs['timestamp'],
                z_camera=meas[:2], verbose=self.verbose)
            self.tracks[new_track.track_id] = new_track

            # Updated matched measurements ID set
            camera_matched_ids.add(int(meas[2]))

        # Update unmatched measurements ID set
        camera_unmatched_ids -= camera_matched_ids

        # Check if all the measurements have been used
        assert len(radar_unmatched_ids) == 0, 'All the radar measurements have not been used'
        assert len(camera_unmatched_ids) == 0, 'All the camera measurements have not been used'

        #--------------------- HANDLE TRACKLET STREAKS ---------------------#

        # Purge old tracklets with no updates
        tracks = {k: v for k,v in self.tracks.iteritems()} 
        for track_id in tracks:
            if tracks[track_id].misses >= self.miss_window:
                if self.verbose:
                    print('Deleting track:', track_id, self.tracks[track_id].state.flatten())
                del self.tracks[track_id]

        # Return confident tracklets
        fused_tracks = {}
        for track_id in self.tracks:
            if self.tracks[track_id].hits >= self.hit_window:
                fused_tracks[track_id] = {}
                fused_tracks[track_id]['state'] = self.tracks[track_id].state.copy().flatten()
                fused_tracks[track_id]['state_cov'] = self.tracks[track_id].state_cov.copy()
                if self.verbose: print('Final states:', track_id, fused_tracks[track_id]['state'])
            elif self.verbose:
                print('Final states (not confident):', track_id, self.tracks[track_id].state.flatten())

        # Store data for the next timestep 
        self.prev_ego_state = inputs['ego_state']
        self.prev_timestamp = inputs['timestamp']

        return fused_tracks


if __name__ == '__main__':    
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
    inputs['camera'] = np.asarray([[1, 2], [3, 4], [0, 0], [5, 6]])
    inputs['radar'] = np.asarray([[1.1, 2, 1, 0], [5.1, 6.2, 0.5, 0.5], [3.1, 4.2, 0, 0], [11, 10, 3, 0]])
    tracker.update(inputs)
