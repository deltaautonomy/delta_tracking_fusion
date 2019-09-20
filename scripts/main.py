#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 07, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Handle paths and OpenCV import
from init_paths import *

# Built-in modules

# External modules
import matplotlib.pyplot as plt

# ROS modules
import tf
import rospy
import message_filters

# ROS messages
from delta_perception.msg import MarkerArrayStamped
from delta_perception.msg import CameraTrack, CameraTrackArray
from radar_msgs.msg import RadarTrack, RadarTrackArray

# Local python modules
from utils import *
from tracker import Tracker

# Global objects
STOP_FLAG = False
cmap = plt.get_cmap('tab10')
tf_listener = None

# Camera variables
CAMERA_INFO = None
CAMERA_EXTRINSICS = None
CAMERA_PROJECTION_MATRIX = None

# Frames
RADAR_FRAME = '/ego_vehicle/radar'
EGO_VEHICLE_FRAME = 'ego_vehicle'
CAMERA_FRAME = 'ego_vehicle/camera/rgb/front'
VEHICLE_FRAME = 'vehicle/%03d/autopilot'

# Perception models
tracker = Tracker()

# FPS loggers
FRAME_COUNT = 0
tracker_fps = FPSLogger('Tracker')



########################### Functions ###########################


def validate(tracks):
    pass


def perception_pipeline(img, radar_msg, publishers, vis=True, **kwargs):
    # Log pipeline FPS
    all_fps.lap()

    # Preprocess
    # img = increase_brightness(img)

    # Object detection
    yolo_fps.lap()
    detections, frame_resized = yolov3.run(img)
    yolo_fps.tick()

    # Display FPS logger status
    all_fps.tick()
    sys.stdout.write('\r%s ' % (tracker_fps.get_log()))
    sys.stdout.flush()

    return detections


def callback(image_msg, radar_msg, objects, publishers, **kwargs):
    # Node stop has been requested
    if STOP_FLAG: return

    # Run the tracking pipeline
    tracks = tracking_pipeline(img.copy(), radar_msg, publishers)

    # Run the validation pipeline
    validate(tracks)


def shutdown_hook():
    global STOP_FLAG
    STOP_FLAG = True
    time.sleep(3)
    print('\n\033[95m' + '*' * 30 + ' Delta Tracking and fusion Shutdown ' + '*' * 30 + '\033[00m\n')
    print ('Tracking results - MOTA:') #TODO

def run(**kwargs):
    global tf_listener, CAMERA_EXTRINSICS

    # Start node
    rospy.init_node('main', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())
    tf_listener = tf.TransformListener()

    # Setup validation
    validation_setup()
    
    # Setup models
    yolov3.setup() 

    # Find the camera to vehicle extrinsics
    tf_listener.waitForTransform(CAMERA_FRAME, RADAR_FRAME, rospy.Time(), rospy.Duration(100.0))
    (trans, rot) = tf_listener.lookupTransform(CAMERA_FRAME, RADAR_FRAME, rospy.Time(0))
    CAMERA_EXTRINSICS = pose_to_transformation(position=trans, orientation=rot)

    # Handle params and topics
    camera_info = rospy.get_param('~camera_info', '/carla/ego_vehicle/camera/rgb/front/camera_info')
    image_color = rospy.get_param('~image_color', '/carla/ego_vehicle/camera/rgb/front/image_color')
    object_array = rospy.get_param('~object_array', '/carla/objects')
    # vehicle_markers = rospy.get_param('~vehicle_markers', '/carla/vehicle_marker_array')
    radar = rospy.get_param('~radar', '/delta/radar/tracks')
    output_image = rospy.get_param('~output_image', '/delta/perception/object_detection_tracking/image')
    camera_track = rospy.get_param('~camera_track', '/delta/perception/camera_track')
    camera_track_marker = rospy.get_param('~camera_track_marker', '/delta/perception/camera_track_marker')
    radar_track_marker = rospy.get_param('~radar_track_marker', '/delta/perception/radar_track_marker')

    # Display params and topics
    rospy.loginfo('CameraInfo topic: %s' % camera_info)
    rospy.loginfo('Image topic: %s' % image_color)
    rospy.loginfo('RADAR topic: %s' % radar)
    rospy.loginfo('CameraTrackArray topic: %s' % camera_track)

    # Publish output topic
    publishers = {}
    publishers['image_pub'] = rospy.Publisher(output_image, Image, queue_size=5)
    publishers['tracker_pub'] = rospy.Publisher(camera_track, CameraTrackArray, queue_size=5)
    publishers['camera_marker_pub'] = rospy.Publisher(camera_track_marker, Marker, queue_size=5)
    publishers['radar_marker_pub'] = rospy.Publisher(radar_track_marker, Marker, queue_size=5)

    # Subscribe to topics
    info_sub = rospy.Subscriber(camera_info, CameraInfo, camera_info_callback)
    image_sub = message_filters.Subscriber(image_color, Image)
    radar_sub = message_filters.Subscriber(radar, RadarTrackArray)
    object_sub = message_filters.Subscriber(object_array, ObjectArray)
    # marker_sub = message_filters.Subscriber(vehicle_markers, MarkerArrayStamped)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, radar_sub, object_sub], queue_size=1, slop=0.5)
    ats.registerCallback(callback, publishers, **kwargs)

    # Shutdown hook
    rospy.on_shutdown(shutdown_hook)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    # Start perception node
    run()
