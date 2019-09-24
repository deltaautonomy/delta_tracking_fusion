#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
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
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from radar_msgs.msg import RadarTrack, RadarTrackArray
from delta_perception.msg import CameraTrack, CameraTrackArray
from delta_prediction.msg import EgoStateEstimate
from delta_tracking_fusion.msg import Track, TrackArray

# Local python modules
from utils import *
from tracker import Tracker
from delta_perception.scripts.cube_marker_publisher import make_label
from delta_perception.scripts.occupancy_grid import OccupancyGridGenerator

# Global objects
STOP_FLAG = False
cmap = plt.get_cmap('tab10')
tf_listener = None

# Frames
RADAR_FRAME = '/ego_vehicle/radar'
EGO_VEHICLE_FRAME = 'ego_vehicle'

# Classes
tracker = Tracker()
occupancy_grid = OccupancyGridGenerator(30, 100, EGO_VEHICLE_FRAME, 0.1)

# FPS loggers
FRAME_COUNT = 0
tracker_fps = FPSLogger('Tracker')


########################### Functions ###########################


def validate(tracks):
    pass


def get_tracker_inputs(camera_msg, radar_msg, state_msg):
    inputs = {'camera': [], 'radar': [], 'ego_state': []}
    inputs['timestamp'] = state_msg.header.stamp
    
    for track in camera_msg.tracks:
        inputs['camera'].append(np.asarray([track.x, track.y]))

    for track in radar_msg.tracks:
        pos_msg = position_to_numpy(track.track_shape.points[0])
        # todo(prateek): trasnform this radar data to ego vehicle frame
        inputs['radar'].append(np.asarray([pos_msg[0] - 2.2, pos_msg[1],
            track.linear_velocity.x, track.linear_velocity.y]))

    inputs['ego_state'] = np.asarray([
        state_msg.pose.position.x,
        state_msg.pose.position.y,
        state_msg.twist.linear.x,
        state_msg.twist.linear.y,
        state_msg.twist.angular.z
    ])

    return inputs

def tracking_fusion_pipeline(camera_msg, radar_msg, state_msg, publishers, vis=True, **kwargs):
    # Log pipeline FPS
    all_fps.lap()

    # Tracker update
    tracker_fps.lap()
    inputs = get_tracker_inputs(camera_msg, radar_msg, state_msg)
    tracks = tracker.update(inputs)
    tracker_fps.tick()

    # Generate ROS messages
    grid = occupancy_grid.empty_grid()
    tracker_array_msg = TrackArray()
    for track_id in tracks:
        state = tracks[track_id]['state']
        state_cov = tracks[track_id]['state_cov']
        # todo: does x, y position for label need to flipped?
        label_msg = make_label('ID: ' + str(track_id), np.r_[state[:2], 1],
            frame_id=EGO_VEHICLE_FRAME, marker_id=track_id)
        grid = occupancy_grid.place_gaussian(state[:2], state_cov[:2, :2], 100, grid)

        # Tracker message
        tracker_msg = Tracker()
        tracker_msg.x = state[0]
        tracker_msg.y = state[1]
        tracker_msg.vx = state[2]
        tracker_msg.vy = state[3]
        tracker_msg.track_id = int(track_id)
        tracker_msg.state_cov = state_cov.flatten().tolist()
        tracker_msg.label = 'vehicle'
        tracker_array_msg.tracks.append(tracker_msg)

    # For debugging without Rviz
    # plt.imshow(grid)
    # plt.show()

    # Publish messages
    grid_msg = occupancy_grid.refresh(grid, radar_msg.header.stamp)
    tracker_array_msg.header.stamp = radar_msg.header.stamp
    publishers['occupancy_pub'].publish(grid_msg)
    publishers['marker_pub'].publish(label_msg)
    publishers['tracker_pub'].publish(tracker_array_msg)

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

    # Handle params and topics
    camera_track = rospy.get_param('~camera_track', '/delta/perception/ipm/camera_track')
    radar_track = rospy.get_param('~radar_track', '/carla/ego_vehicle/radar/tracks')
    ego_state = rospy.Publisher('~ego_state', '/delta/prediction/ego_vehicle/state')
    fused_track = rospy.get_param('~fused_track', '/delta/tracking_fusion/tracker/tracks')
    occupancy_topic = rospy.get_param('~occupancy_topic', '/delta/tracking_fusion/tracker/occupancy_grid')
    track_marker = rospy.get_param('~track_marker', '/delta/tracking_fusion/tracker/track_id_marker')

    # Display params and topics
    rospy.loginfo('CameraTrackArray topic: %s' % camera_track)
    rospy.loginfo('RadarTrackArray topic: %s' % radar_track)
    rospy.loginfo('EgoStateEstimate topic: %s' % ego_state)
    rospy.loginfo('TrackArray topic: %s' % fused_track)
    rospy.loginfo('OccupancyGrid topic: %s' % occupancy_topic)
    rospy.loginfo('Track ID Marker topic: %s' % track_marker)

    # Publish output topic
    publishers = {}
    publishers['track_pub'] = rospy.Publisher(fused_track, TrackArray, queue_size=5)
    publishers['occupancy_pub'] = rospy.Publisher(occupancy_topic, OccupancyGrid, queue_size=5)
    publishers['marker_pub'] = rospy.Publisher(track_marker, Marker, queue_size=5)

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