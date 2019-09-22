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

# ROS modules
import rospy
import message_filters

# ROS messages
from radar_msgs.msg import RadarTrack, RadarTrackArray
from delta_perception.msg import CameraTrack, CameraTrackArray
from delta_prediction.msg import EgoStateEstimate
from delta_tracking_fusion.msg import FusedTrack, FusedTrackArray

# Local python modules
from utils import *
from tracker import Tracker

# Global objects
STOP_FLAG = False
cmap = plt.get_cmap('tab10')
tf_listener = None

# Frames
RADAR_FRAME = '/ego_vehicle/radar'
EGO_VEHICLE_FRAME = 'ego_vehicle'


# FPS loggers
FRAME_COUNT = 0
tracker_fps = FPSLogger('Tracker')

# Global thresholds
HIT_THRESHOLD = 1
MISS_THRESHOLD = 3
R_radar = np.array([[10,  0,  0,  0],
              [ 0, 10,  0,  0], 
              [ 0,  0, 10,  0],
              [ 0,  0,  0, 10]])

# Classes
mot_tracker = Tracker(HIT_THRESHOLD, MISS_THRESHOLD, R_radar)

########################### Functions ###########################


def validate(tracks):
    pass


def get_tracker_inputs(camera_msg, radar_msg, state_msg):
    inputs = {'camera_measurement': [], 'radar_measurement': [], 'ego_state': []}
    inputs['timestamp'] = state_msg.header.stamp
    
    for track in camera_msg.tracks:
        inputs['camera_measurement'].append(np.asarray([track.x, track.y]))
    
    for track in radar_msg.tracks:
        pos_msg = position_to_numpy(track.track_shape.points[0])
        # todo(prateek): trasnform this radar data to ego vehicle frame
        inputs['radar_measurement'].append(np.asarray([pos_msg[0] - 2.2, pos_msg[1],
            track.linear_velocity.x, track.linear_velocity.y]))

    inputs['ego_state'] = np.asarray([
        state_msg.pose.position.x,
        state_msg.pose.position.y,
        ego_state.twist.linear.x,
        ego_state.twist.linear.y,
    ])

    return inputs

def tracking_fusion_pipeline(camera_msg, radar_msg, state_msg, publishers, vis=True, **kwargs):
    # Log pipeline FPS
    all_fps.lap()

    # Tracking initialization 
    tracker_fps.lap()

    # update step on mot_tracker
    inputs = get_tracker_inputs(camera_msg, radar_msg, state_msg)
    tracks = mot_tracker.step(inputs)

    tracker_fps.tick()

    # Display FPS logger status
    all_fps.tick()
    sys.stdout.write('\r%s ' % (tracker_fps.get_log()))
    sys.stdout.flush()

    return tracks


def callback(camera_msg, radar_msg, state_msg, publishers, **kwargs):
    # Node stop has been requested
    if STOP_FLAG: return

    # Run the tracking pipeline
    tracks = tracking_fusion_pipeline(camera_msg, radar_msg, state_msg, publishers)

    # Run the validation pipeline
    validate(tracks)


def shutdown_hook():
    global STOP_FLAG
    STOP_FLAG = True
    time.sleep(3)
    print('\n\033[95m' + '*' * 30 + ' Delta Tracking and Fusion Shutdown ' + '*' * 30 + '\033[00m\n')
    # print('Tracking results - MOTA:') #TODO


def run(**kwargs):
    # Start node
    rospy.init_node('tracking_fusion_pipeline', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())

    # Setup validation
    validation_setup()

    # Handle params and topics
    camera_track = rospy.get_param('~camera_track', '/delta/perception/ipm/camera_track')
    radar_track = rospy.get_param('~radar_track', '/carla/ego_vehicle/radar/tracks')
    ego_state = rospy.Publisher('~ego_state', '/delta/prediction/ego_vehicle/state')
    fused_track = rospy.get_param('~radar_track', '/delta/tracking_fusion/tracker/tracks')

    # Display params and topics
    rospy.loginfo('CameraTrackArray topic: %s' % camera_track)
    rospy.loginfo('RadarTrackArray topic: %s' % radar_track)
    rospy.loginfo('EgoStateEstimate topic: %s' % ego_state)
    rospy.loginfo('FusedTrackArray topic: %s' % fused_track)

    # Publish output topic
    publishers = {}
    publishers['track_pub'] = rospy.Publisher(fused_track, FusedTrackArray, queue_size=5)

    # Subscribe to topics
    camera_sub = message_filters.Subscriber(camera_track, CameraTrackArray)
    radar_sub = message_filters.Subscriber(radar_track, RadarTrackArray)
    state_sub = message_filters.Subscriber(ego_state, EgoStateEstimate)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [camera_sub, radar_sub, state_sub], queue_size=1, slop=0.5)
    ats.registerCallback(callback, publishers, **kwargs)

    # Shutdown hook
    rospy.on_shutdown(shutdown_hook)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    # Start tracking fusion node
    run()