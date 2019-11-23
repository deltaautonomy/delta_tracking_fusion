#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Nov 22, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Handle paths and OpenCV import
from init_paths import *

# External modules
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# ROS modules
import rospy
import message_filters

# ROS messages
from delta_msgs.msg import Track, TrackArray
from radar_msgs.msg import RadarTrack, RadarTrackArray

# Local python modules
from utils import *

# Global objects
STOP_FLAG = False

# Frames
RADAR_FRAME = '/ego_vehicle/radar'
EGO_VEHICLE_FRAME = 'ego_vehicle'

np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)


########################### Functions ###########################


def data_association(states_a, states_b, gating_threshold=5):
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


def convert_message_to_numpy(msg):   
    tracks = [[track.x, track.y, track.vx, track.vy] for track in msg.tracks]
    return np.array(tracks)


def convert_radar_to_numpy(msg):
    tracks = []
    for track in msg.tracks:
        pos = position_to_numpy(track.track_shape.points[0])
        tracks.append([pos[0] - 2.2, pos[1],
            track.linear_velocity.x, track.linear_velocity.y])
    return np.array(tracks)


def callback(gt_msg, tracks_msg, radar_msg, **kwargs):
    # Node stop has been requested
    if STOP_FLAG: return

    # Run the validation pipeline
    print('------------------------')
    ground_truth = convert_message_to_numpy(gt_msg)
    tracks = convert_message_to_numpy(tracks_msg)
    radar_tracks = convert_radar_to_numpy(radar_msg)
    print('GT')
    print(ground_truth)
    print('Radar')
    print(radar_tracks)
    print('Pred')
    print(tracks)

    # args = data_association(ground_truth, tracks)
    # print(args)


def shutdown_hook():
    global STOP_FLAG
    STOP_FLAG = True
    time.sleep(3)

    print('\n\033[95m' + '*' * 30 + ' Velocity Validation ' + '*' * 30 + '\033[00m\n')


def run(**kwargs):
    # Start node
    rospy.init_node('tracking_fusion_pipeline', anonymous=False)
    rospy.loginfo('Current PID: [%d]' % os.getpid())

    # Handle params and topics
    ground_truth_track = rospy.get_param('~ground_truth_track', '/carla/ego_vehicle/tracks/ground_truth')
    track_array = rospy.get_param('~track_array', '/delta/tracking_fusion/tracker/tracks')
    radar_track = rospy.get_param('~radar_track', '/carla/ego_vehicle/radar/tracks')

    # Subscribe to topics
    ground_truth_sub = message_filters.Subscriber(ground_truth_track, TrackArray)
    track_sub = message_filters.Subscriber(track_array, TrackArray)
    radar_sub = message_filters.Subscriber(radar_track, RadarTrackArray)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [ground_truth_sub, track_sub, radar_sub], queue_size=1, slop=0.5)
    ats.registerCallback(callback)

    # Shutdown hook
    rospy.on_shutdown(shutdown_hook)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    run()
