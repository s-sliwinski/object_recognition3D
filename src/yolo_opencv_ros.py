#!/usr/bin/python

"""Node for publishing bboxes(with yolov4 and openCV) for further use"""

from __future__ import print_function, with_statement
import os
import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
CLASS_NAMES = []
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
ENABLE_CUDA = False

class YoloBbox:
    def __init__(self):
        rospy.loginfo("detector is running...")
        # frame size
        self.frame_x = 416
        self.frame_y = 416

        self.bridge = CvBridge()

        # preapare detector with OpenCV
        self.net = cv2.dnn.readNet("src/object_recognition3D/src/yolo_utils/yolov4.weights", "src/object_recognition3D/src/yolo_utils/yolov4.cfg")
        if ENABLE_CUDA:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(self.frame_x, self.frame_y), scale=1/255)

        # init Publisher and subscriber
        self.img_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback)
        self.img_detected_pub = rospy.Publisher("/yolo_img_detected_frame", Image, queue_size=100)

    def self.rgb_callback(data):
        pass

def read_class_name():
    #long path for using with rosrun
    with open("src/object_recognition3D/src/yolo_utils/coco.names", "r") as f:
        CLASS_NAMES = [class_name.strip() for class_name in f.readlines()]
def main():
    read_class_name()
    rospy.init_node('yolo_opencv', anonymous=True)
    ybb = YoloBbox()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("...shutting down")

if __name__ == "__main__":
    main()
