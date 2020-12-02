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
        # setup markers
        self.marker_array = MarkerArray()
        self.markers_count = 0
        self.MARKERS_MAX = 3

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
        self.marker_array_pub= rospy.Publisher("/img_detected_markers", MarkerArray, queue_size=100)
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


    def construct_marker(self, p_coord):
        # position extracted from points_coordinates array
        if  len(p_coord[0]) != 0:
            marker = Marker()
            marker.header.frame_id = "camera_link"
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # try to calculate x and y scale
            if len(p_coord[1]) != 0:
                marker.scale.x = abs(p_coord[0][0][2] - p_coord[1][0][2]) * 2 # abs(center_z - left_upper_z) *2
                marker.scale.y = abs(p_coord[0][0][0] - p_coord[1][0][0]) * 2 # abs(center_x - left_upper_x) *2
                marker.scale.z = abs(p_coord[0][0][1] - p_coord[1][0][1]) * 2 # abs(center_y - left_upper_y) *2
            elif len(p_coord[2]) != 0:
                marker.scale.x = abs(p_coord[0][0][2] - p_coord[2][0][2]) * 2 # abs(center_z - right_lower_z) *2
                marker.scale.y = abs(p_coord[0][0][0] - p_coord[2][0][0]) * 2 # abs(center_x - right_lower_x) *2
                marker.scale.z = abs(p_coord[0][0][1] - p_coord[2][0][1]) * 2 # abs(center_y - right_lower_y) *2
            elif len(p_coord[3]) != 0:
                marker.scale.x = abs(p_coord[0][0][2] - p_coord[3][0][2]) * 2 # abs(center_z - right_upper_z) *2
                marker.scale.y = abs(p_coord[0][0][0] - p_coord[3][0][0]) * 2 # abs(center_x - right_upper_x) *2
                marker.scale.z = abs(p_coord[0][0][1] - p_coord[3][0][1]) * 2 # abs(center_y - right_upper_y) *2
            elif len(p_coord[4]) != 0:
                marker.scale.x = abs(p_coord[0][0][2] - p_coord[4][0][2]) * 2 # abs(center_z - left_lower_z) *2
                marker.scale.y = abs(p_coord[0][0][0] - p_coord[4][0][0]) * 2 # abs(center_x - left_lower_x) *2
                marker.scale.z = abs(p_coord[0][0][1] - p_coord[4][0][1]) * 2 # abs(center_y - left_lower_y) *2
            # take default size
            else:
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2

            marker.color.a = 0.5
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0

            # reverse cooridinates to match /camera_link
            marker.pose.position.x = p_coord[0][0][2] #z
            marker.pose.position.y = -p_coord[0][0][0] #x
            marker.pose.position.z = -p_coord[0][0][1] #y

            if self.markers_count > self.MARKERS_MAX:
                self.marker_array.markers.pop(0)

            self.marker_array.markers.append(marker)

            id = 0
            for m in self.marker_array.markers:
                m.id = id
                id += 1

            self.markers_count += 1

    # main functionality publishing detected bouning box img and doing transform
    def detection(self):

        for bbox in self.detected_objects:
            xA, yA, xB, yB = bbox
            x_center = xB - ((xB - xA) // 2)
            y_center = yB - ((yB - yA) // 2)
            # draw rectangle for testing
            cv2.rectangle(self.cv_image, (xA, yA), (xB, yB), (0,255,0), 3)

            # get points of detected object
            point_coordinates = []
            point_coordinates.append(list(pc2.read_points(self.pcl, skip_nans=True, field_names=('x', 'y', 'z'), uvs=[(x_center, y_center)])))
            point_coordinates.append(list(pc2.read_points(self.pcl, skip_nans=True, field_names=('x', 'y', 'z'), uvs=[(xA, yA)])))
            point_coordinates.append(list(pc2.read_points(self.pcl, skip_nans=True, field_names=('x', 'y', 'z'), uvs=[(xB, yB)])))
            point_coordinates.append(list(pc2.read_points(self.pcl, skip_nans=True, field_names=('x', 'y', 'z'), uvs=[(xB, yA)])))
            point_coordinates.append(list(pc2.read_points(self.pcl, skip_nans=True, field_names=('x', 'y', 'z'), uvs=[(xA, yB)])))

            # reverse coordinatse to match /camera_link
            point_coordinates_reversed = []
            print (point_coordinates)
            if len(point_coordinates) > 0:
                self.construct_marker(point_coordinates)
                for point_list in point_coordinates:
                    if len(point_list) != 0:
                        x_point, y_point, z_point = point_list[0]
                        point_list[0] = (z_point, -x_point, -y_point)
                        point_coordinates_reversed.append(point_list)
                # publish reversed transformations
                print(point_coordinates_reversed)
                for i in range((len(point_coordinates_reversed))):
                    self.tf_pub.sendTransform((point_coordinates_reversed[i][0]), q(0,0,0), rospy.Time.now(), 'object' + str(bbox[0]) + str(i), 'camera_link')
        try:
            img_msg = self.bridge.cv2_to_imgmsg(self.cv_image, 'bgr8')
        except CvBridgeError as e:
            print(e)

        # publish image with bbox
        self.img_detected_pub.publish(img_msg)
        # publish markers array
        self.marker_array_pub.publish(self.marker_array)

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
