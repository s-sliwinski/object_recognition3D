#!/usr/bin/python

""" Node for producing point cloud from RGB and Depth image """

from __future__ import print_function
import rospy
import tf
from tf.transformations import quaternion_from_euler as q
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class MakePointCloud:
    def __init__(self):
        rospy.loginfo("pointcloud object detection is running...")

        # frame size
        self.frame_x = 640
        self.frame_y = 480

        self.bridge = CvBridge()

        # mockup detected objects
        self.detected_objects = [[(self.frame_x // 2 - 50), (self.frame_y // 2 - 50), (self.frame_x//2 + 50), (self.frame_y // 2 + 50)],
                                [20, 20, 80, 80],
                                [500, 300, 600, 400]]

        # cv_image and pcl variables
        self.cv_image = np.zeros([self.frame_x, self.frame_y])
        self.pcl = None

        # transform config
        self.tf_pub = tf.TransformBroadcaster()

        # subscribers
        self.img_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback)
        self.pcl_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pcl_callback)
        # publishers
        self.img_detected_pub = rospy.Publisher("/img_detected_frame", Image, queue_size=100)
        self.rate = rospy.Rate(1)

    # callback converting rgb to cv2_image
    def rgb_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
        frame_resized = cv2.resize(frame, (self.frame_x, self.frame_y))
        self.cv_image = frame_resized

    # callback for doing something with incoming pointcloud and running detection
    def pcl_callback(self, data):
        self.pcl = data
        self.detection()

    # main functionality publishing detected bouning box img and doing transform
    def detection(self):

        for bbox in self.detected_objects:
            xA, yA, xB, yB = bbox
            x_center = xB - ((xB - xA) // 2)
            y_center = yB - ((yB - yA) // 2)
            # draw rectangle for testing
            cv2.rectangle(self.cv_image, (xA, yA), (xB, yB), (0,255,0), 3)

            # get points of detected object
            point_coordinates = list(pc2.read_points(self.pcl, skip_nans=True, field_names=('x', 'y', 'z'), uvs=[(xA, yA),(xB, yB),(x_center, y_center),(xB, yA),(xA, yB)]))

            # reverse coordinatse to match /camera_link
            point_coordinates_reversed = []
            print (point_coordinates)
            if len(point_coordinates) > 0:
                for point_list in point_coordinates:
                    if len(point_list) != 0:
                        x_point, y_point, z_point = point_list
                        point_list = (z_point, -x_point, -y_point)
                        point_coordinates_reversed.append(point_list)
                # publish reversed transformations
                for i in range((len(point_coordinates_reversed))):
                    self.tf_pub.sendTransform((point_coordinates_reversed[i]), q(0,0,0), rospy.Time.now(), 'object' + str(bbox[0]) + str(i), 'camera_link')
        try:
            img_msg = self.bridge.cv2_to_imgmsg(self.cv_image, 'bgr8')
        except CvBridgeError as e:
            print(e)

        # publish image with bbox
        self.img_detected_pub.publish(img_msg)
        #self.rate.sleep()
def main():
    rospy.init_node('make_point_cloud', anonymous=True)
    mpc = MakePointCloud()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("...shutting down")

if __name__ == "__main__":
    main()
