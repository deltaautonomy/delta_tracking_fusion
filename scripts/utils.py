#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 07, 2019
'''

# Handle paths and OpenCV import
from init_paths import *

# External modules
import PIL.Image

# ROS modules
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from diagnostic_msgs.msg import DiagnosticStatus
from tf.transformations import (translation_matrix,
                                quaternion_matrix,
                                concatenate_matrices,
                                euler_from_quaternion)

# Global variables
CV_BRIDGE = CvBridge()


########################### Functions ###########################


class FPSLogger:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.fps = None
        self.last = None
        self.total_time = 0
        self.total_frames = 0

    def lap(self):
        self.last = time.time()

    def tick(self, count=1):
        self.total_time += time.time() - self.last
        self.total_frames += count
        self.fps = self.total_frames / self.total_time

    def log(self, tick=False):
        if tick: self.tick()
        print('\033[94m %s FPS:\033[00m \033[93m%.1f\033[00m' % (self.name, self.fps))

    def get_log(self, tick=False):
        if tick: self.tick()
        return '\033[94m %s FPS:\033[00m \033[93m%.1f\033[00m' % (self.name, self.fps)


def pil_image(img):
    return PIL.Image.fromarray(img)


def cv_image(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def message_to_cv2(msg):
    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(msg, 'bgr8')
        return img
    except CvBridgeError as e:
        print(e)
        rospy.logerr(e)
        return None


def cv2_to_message(img, topic):
    # Publish image using CV bridge
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    try:
        topic.publish(CV_BRIDGE.cv2_to_imgmsg(img, "bgr8"))
    except CvBridgeError as e: 
        print(e)
        rospy.logerr(e)


def position_to_numpy(position):
    return np.asarray([position.x, position.y, position.z])


def orientation_to_numpy(orientation):
    return np.asarray([orientation.x, orientation.y, orientation.z, orientation.w])


def orientation_to_rpy(orientation):
    return euler_from_quaternion(orientation_to_numpy(quaternion))


def quaternion_to_rpy(quaternion):
    return euler_from_quaternion(quaternion)


def pose_to_transformation(pose=None, position=None, orientation=None):
    if position is None:
        position = position_to_numpy(pose.position)
    if orientation is None:
        orientation = orientation_to_numpy(pose.orientation)
    return concatenate_matrices(translation_matrix(position), quaternion_matrix(orientation))


def make_diagnostics_status(name, pipeline, fps, level=DiagnosticStatus.OK):
    msg = DiagnosticStatus()
    msg.level = DiagnosticStatus.OK
    msg.name = name
    msg.message = fps
    msg.hardware_id = pipeline
    return msg
