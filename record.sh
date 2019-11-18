#!/bin/bash

rosbag record \
    /carla/ego_vehicle/ackermann_cmd \
    /carla/ego_vehicle/camera/rgb/front/camera_info \
    /carla/ego_vehicle/camera/rgb/front/image_color \
    /carla/ego_vehicle/ego_vehicle_control_info \
    /carla/ego_vehicle/gnss/front/gnss \
    /carla/ego_vehicle/odometry \
    /carla/ego_vehicle/parameter_descriptions \
    /carla/ego_vehicle/parameter_updates \
    /carla/ego_vehicle/radar/tracks \
    /carla/ego_vehicle/tracks/ground_truth \
    /carla/map \
    /carla/objects \
    /carla/vehicle_marker \
    /clock \
    /delta/perception/camera_track_marker \
    /delta/perception/ipm/camera_track \
    /delta/perception/object_detection_tracking/image \
    /delta/perception/occupancy_grid \
    /delta/perception/radar_track_marker \
    /delta/prediction/ego \
    /delta/prediction/ego_vehicle/state \
    /delta/prediction/ego_vehicle/trajectory \
    /delta/prediction/ground_truth \
    /image_view_input/output \
    /image_view_input/parameter_descriptions \
    /image_view_input/parameter_updates \
    /rosout \
    /rosout_agg \
    /tf \
    /tf_static
