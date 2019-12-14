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
import pprint

# External modules
import motmetrics as mot
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ROS modules
import rospy
import message_filters

# ROS messages
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from jsk_rviz_plugins.msg import PictogramArray
from diagnostic_msgs.msg import DiagnosticArray
from radar_msgs.msg import RadarTrack, RadarTrackArray
from delta_msgs.msg import (CameraTrack,
                            CameraTrackArray,
                            EgoStateEstimate,
                            Track,
                            TrackArray)

# Local python modules
from utils import *
from tracker import Tracker
from scripts.occupancy_grid import OccupancyGridGenerator
from scripts.cube_marker_publisher import make_label, make_pictogram, make_trajectory

# Global objects
STOP_FLAG = False
cmap = plt.get_cmap('tab20')
cmap_colors = [cmap(i) for i in range(20) if i % 2 != 0]
tf_listener = None
trajectories = {}

# Frames
RADAR_FRAME = '/ego_vehicle/radar'
EGO_VEHICLE_FRAME = 'ego_vehicle'

# Classes
pp = pprint.PrettyPrinter(indent=4)
tracker = Tracker()
acc = mot.MOTAccumulator(auto_id=True)
occupancy_grid = OccupancyGridGenerator(30, 100, EGO_VEHICLE_FRAME, 0.5)

# FPS loggers
FRAME_COUNT = 0
tracker_fps = FPSLogger('Tracker')
all_fps = FPSLogger('All')


########################### Functions ###########################


def validate(tracks, gt_msg, max_sq_dist=100.0):
    # Compute cost matrix.
    detections = np.asarray([tracks[track_id]['state'][:2] for track_id in tracks])
    ground_truth = np.asarray([[track.x, track.y] for track in gt_msg.tracks])
    cost_matrix = mot.distances.norm2squared_matrix(ground_truth, detections, max_d2=max_sq_dist)

    # Accumulate data for validation.
    gt_labels = [track.track_id for track in gt_msg.tracks]
    acc.update(gt_labels, tracks.keys(), cost_matrix)


def validate_iou(tracks, gt_msg, bbox_dim=[10.0, 10.0], max_iou=0.5):
    # Compute cost matrix.
    detections = np.asarray([np.r_[tracks[track_id]['state'][:2], bbox_dim] for track_id in tracks])
    ground_truth = np.asarray([np.r_[[track.x, track.y], bbox_dim] for track in gt_msg.tracks])
    cost_matrix = mot.distances.iou_matrix(ground_truth, detections, max_iou=max_iou)

    # Accumulate data for validation.
    gt_labels = [track.track_id for track in gt_msg.tracks]
    acc.update(gt_labels, tracks.keys(), cost_matrix)


def make_track_msg(track_id, state, state_cov, ego_state):
    tracker_msg = Track()
    tracker_msg.x = state[0]
    tracker_msg.y = state[1]
    tracker_msg.vx = state[2] + ego_state.twist.linear.x
    tracker_msg.vy = state[3] + ego_state.twist.linear.y
    tracker_msg.track_id = int(track_id)
    tracker_msg.covariance = state_cov.flatten().tolist()
    tracker_msg.label = 'vehicle'
    return tracker_msg


def publish_trajectory(publishers, track_id, state, tracks, smoothing=True, max_length=35):
    global trajectories

    # Create/update trajectory
    if track_id not in trajectories: trajectories[track_id] = np.asarray([state[:2]])
    else: trajectories[track_id] = np.append(trajectories[track_id], [state[:2]], axis=0)

    # Trajectory smoothing over time
    if smoothing:
        length = len(trajectories[track_id])
        if length > 5:
            poly_degree = 4
            window = int(min(np.ceil(length - 2) // 2 * 2 + 1, 51))
            trajectories[track_id][:, 0] = savgol_filter(trajectories[track_id][:, 0], window, poly_degree)
            trajectories[track_id][:, 1] = savgol_filter(trajectories[track_id][:, 1], window, poly_degree)

    # Publish the trajectory
    publishers['traj_pub'].publish(make_trajectory(trajectories[track_id][-max_length:],
        frame_id=EGO_VEHICLE_FRAME, marker_id=track_id, color=cmap_colors[track_id % 10]))


def publish_messages(publishers, tracks, ego_state, timestamp):
    global trajectories

    # Generate ROS messages
    grid = occupancy_grid.empty_grid()
    tracker_array_msg = TrackArray()
    label_array_msg = PictogramArray()

    for track_id in tracks:
        # Occupancy grid
        state = tracks[track_id]['state']
        state_cov = tracks[track_id]['state_cov']
        grid = occupancy_grid.place(state[:2], 100, grid)
        # grid = occupancy_grid.place_gaussian(state[:2], state_cov[:2, :2], 100, grid)

        # Text marker
        publishers['marker_pub'].publish(make_label('ID ' + str(track_id),
            np.r_[state[:2], 1], frame_id=EGO_VEHICLE_FRAME, marker_id=track_id))

        # Label icon
        label_array_msg.pictograms.append(make_pictogram('fa-car',
            np.r_[state[:2], 3], frame_id=EGO_VEHICLE_FRAME))

        # Tracker message
        tracker_array_msg.tracks.append(make_track_msg(track_id, state, state_cov, ego_state))

        # Update and publish trajectory
        publish_trajectory(publishers, track_id, state.copy(), tracks)

    # Publish messages
    grid_msg = occupancy_grid.refresh(grid, timestamp)
    publishers['occupancy_pub'].publish(grid_msg)

    label_array_msg.header.stamp = rospy.Time.now()
    label_array_msg.header.frame_id = EGO_VEHICLE_FRAME
    publishers['label_pub'].publish(label_array_msg)

    tracker_array_msg.header.stamp = timestamp
    tracker_array_msg.header.frame_id = EGO_VEHICLE_FRAME
    publishers['track_pub'].publish(tracker_array_msg)

    # Purge old trajectories
    trajectories_ = {k: v for k,v in trajectories.iteritems()} 
    for track_id in trajectories_:
        if track_id not in tracks: del trajectories[track_id]


def get_tracker_inputs(camera_msg, radar_msg, state_msg):
    inputs = {'camera': [], 'radar': [], 'ego_state': []}
    inputs['timestamp'] = state_msg.header.stamp.to_sec()

    for track in camera_msg.tracks:
        inputs['camera'].append(np.asarray([track.x, track.y]))
    inputs['camera'] = np.asarray(inputs['camera'])

    for track in radar_msg.tracks:
        pos_msg = position_to_numpy(track.track_shape.points[0])
        inputs['radar'].append(np.asarray([pos_msg[0] - 2.2, pos_msg[1],
            track.linear_velocity.x, track.linear_velocity.y]))
    inputs['radar'] = np.asarray(inputs['radar'])

    inputs['ego_state'] = np.asarray([
        state_msg.pose.position.x,
        state_msg.pose.position.y,
        state_msg.twist.linear.x,
        state_msg.twist.linear.y,
        state_msg.twist.angular.z
    ])

    return inputs


def publish_diagnostics(publishers):
    msg = DiagnosticArray()
    msg.status.append(make_diagnostics_status('tracker', 'tracking_fusion', str(tracker_fps.fps)))
    publishers['diag_pub'].publish(msg)


def tracking_fusion_pipeline(camera_msg, radar_msg, state_msg,
    publishers, vis=True, **kwargs):
    # Tracker update
    tracker_fps.lap()
    inputs = get_tracker_inputs(camera_msg, radar_msg, state_msg)
    tracks = tracker.update(inputs)
    tracker_fps.tick()

    # Publish all messages
    publish_messages(publishers, tracks, state_msg, timestamp=radar_msg.header.stamp)

    # Display FPS logger status
    # sys.stdout.write('\r%s ' % (tracker_fps.get_log()))
    # sys.stdout.flush()

    # Publish diagnostics status
    publish_diagnostics(publishers)

    return tracks


def callback(camera_msg, radar_msg, state_msg, gt_msg, publishers, **kwargs):
    # Node stop has been requested
    if STOP_FLAG: return

    # Run the tracking pipeline
    tracks = tracking_fusion_pipeline(camera_msg, radar_msg, state_msg, publishers)

    # Run the validation pipeline
    validate_iou(tracks, gt_msg)


def shutdown_hook():
    global STOP_FLAG
    STOP_FLAG = True
    time.sleep(3)

    print('\n\033[95m' + '*' * 30 + ' Delta Tracking and Fusion Shutdown ' + '*' * 30 + '\033[00m\n')
    # print('\n\033[95m' + '*' * 30 + ' MOT Events Summary ' + '*' * 30 + '\033[00m\n')
    # print(acc.mot_events)

    # Compute and display tracking metrics
    print('\n\033[95m' + '*' * 30 + ' MOT Metrics Summary ' + '*' * 30 + '\033[00m\n')
    metrics = mot.metrics.create()
    summary = metrics.compute(acc, metrics=mot.metrics.motchallenge_metrics, name='Overall')
    print(mot.io.render_summary(summary, formatters=metrics.formatters,
        namemap=mot.io.motchallenge_metric_names), '\n')


def run(**kwargs):
    # Start node
    rospy.init_node('tracking_fusion_pipeline', anonymous=False)
    rospy.loginfo('Current PID: [%d]' % os.getpid())

    # Handle params and topics
    camera_track = rospy.get_param('~camera_track', '/delta/perception/ipm/camera_track')
    radar_track = rospy.get_param('~radar_track', '/carla/ego_vehicle/radar/tracks')
    ground_truth_track = rospy.get_param('~ground_truth_track', '/carla/ego_vehicle/tracks/ground_truth')
    ego_state = rospy.get_param('~ego_state', '/delta/prediction/ego_vehicle/state')
    fused_track = rospy.get_param('~fused_track', '/delta/tracking_fusion/tracker/tracks')
    occupancy_topic = rospy.get_param('~occupancy_topic', '/delta/tracking_fusion/tracker/occupancy_grid')
    track_marker = rospy.get_param('~track_marker', '/delta/tracking_fusion/tracker/track_id_marker')
    label_marker = rospy.get_param('~label_marker', '/delta/tracking_fusion/tracker/label_marker')
    trajectory_marker = rospy.get_param('~trajectory_marker', '/delta/tracking_fusion/tracker/trajectory_marker')
    diagnostics = rospy.get_param('~diagnostics', '/delta/tracking_fusion/tracker/diagnostics')

    # Display params and topics
    rospy.loginfo('CameraTrackArray topic: %s' % camera_track)
    rospy.loginfo('RadarTrackArray topic: %s' % radar_track)
    rospy.loginfo('Ground Trurh TrackArray topic: %s' % ground_truth_track)
    rospy.loginfo('EgoStateEstimate topic: %s' % ego_state)
    rospy.loginfo('TrackArray topic: %s' % fused_track)
    rospy.loginfo('OccupancyGrid topic: %s' % occupancy_topic)
    rospy.loginfo('Track ID Marker topic: %s' % track_marker)
    rospy.loginfo('Label Marker topic: %s' % label_marker)
    rospy.loginfo('Trajectory Marker topic: %s' % trajectory_marker)
    rospy.loginfo('Diagnostics topic: %s' % diagnostics)

    # Publish output topic
    publishers = {}
    publishers['track_pub'] = rospy.Publisher(fused_track, TrackArray, queue_size=5)
    publishers['occupancy_pub'] = rospy.Publisher(occupancy_topic, OccupancyGrid, queue_size=5)
    publishers['marker_pub'] = rospy.Publisher(track_marker, Marker, queue_size=5)
    publishers['label_pub'] = rospy.Publisher(label_marker, PictogramArray, queue_size=5)
    publishers['traj_pub'] = rospy.Publisher(trajectory_marker, Marker, queue_size=5)
    publishers['diag_pub'] = rospy.Publisher(diagnostics, DiagnosticArray, queue_size=5)

    # Subscribe to topics
    camera_sub = message_filters.Subscriber(camera_track, CameraTrackArray)
    radar_sub = message_filters.Subscriber(radar_track, RadarTrackArray)
    ground_truth_sub = message_filters.Subscriber(ground_truth_track, TrackArray)
    state_sub = message_filters.Subscriber(ego_state, EgoStateEstimate)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [camera_sub, radar_sub, state_sub, ground_truth_sub], queue_size=1, slop=0.5)
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
    rospy.init_node('tracking_fusion_pipeline')
    run()
