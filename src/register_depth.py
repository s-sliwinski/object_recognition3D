#!/usr/bin/python

""" Node for producing point cloud from RGB and Depth image """

from __future__ import print_function
import rospy
import tf2_ros as tf
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2
from std_msgs.msg import Header

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class MakePointCloud:
    def __init__(self):
        rospy.loginfo("pointcloud object detection is running...")

        # subscribers
        self.img_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback)
        self.pcl_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pcl_callback)
        # publishers
        self.img_detected_pub = rospy.Publisher("/img_detected_frame", Image, queue_size=100)

        # frame size
        self.frame_x = 640
        self.frame_y = 480

        self.bridge = CvBridge()

        # cv_image and pcl variables
        self.cv_image = None
        self.pcl = None

        # transform config
        tfBuffer = tf.Buffer()
        self.tf_listener = tf.TransformListener(tfBuffer)
        self.tf_pub = tf.TransformBroadcaster()

    # callback converting rgb to cv2_image
    def rgb_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
        frame_resized = cv2.resize(frame, (self.frame_x, self.frame_y))
        self.cv_image = frame_resized

    # callback for doing something with coming pointcloud and running detection
    def pcl_callback(self, data):
        self.pcl = data
        self.detection()

    # main functionality publishing detected bouning box img and doing transform
    def detection(self):
        # draw rectangle 100x100 in the center for testing
        cv2.rectangle(self.cv_image, ((self.frame_x // 2 - 50), (self.frame_y // 2 - 50)), ((self.frame_x//2 + 50), (self.frame_y // 2 + 50)), (0,255,0), 3)

        try:
            img_msg = self.bridge.cv2_to_imgmsg(self.cv_image, 'bgr8')
        except CvBridgeError as e:
            print(e)

        self.img_detected_pub.publish(img_msg)







def main():
    rospy.init_node('make_point_cloud', anonymous=True)
    mpc = MakePointCloud()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("...shutting down")

if __name__ == "__main__":
    main()
