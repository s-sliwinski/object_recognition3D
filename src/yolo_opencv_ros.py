#!/usr/bin/python

"""Node for publishing bboxes(with yolov4 and openCV) for further use"""

from __future__ import print_function, with_statement
import os
import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
CLASS_NAMES = []
# COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
COLORS = []
ENABLE_CUDA = False

class YoloBbox:
    def __init__(self):
        rospy.loginfo("detector is running...")
        # frame size
        self.frame_x = 416
        self.frame_y = 416

        self.bridge = CvBridge()

        # preapare detector with OpenCV
        self.net = cv2.dnn.readNetFromDarknet("src/object_recognition3D/src/yolo_utils/yolov3-tiny.cfg", "src/object_recognition3D/src/yolo_utils/yolov3-tiny.weights")
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        if ENABLE_CUDA:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        # self.model = cv2.dnn_DetectionModel(self.net)
        # self.model.setInputParams(size=(self.frame_x, self.frame_y), scale=1/255)

        # init Publisher and subscriber
        self.img_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback)
        self.img_detected_pub = rospy.Publisher("/yolo_img_detected_frame", Image, queue_size=100)


    def rgb_callback(self, img):
        # convert from IMG_MSG to CV2 format
        try:
            frame = self.bridge.imgmsg_to_cv2(img, 'bgr8')
        except CvBridgeError as e:
            print(e)
        frame = cv2.resize(frame, (self.frame_x, self.frame_y))
        (W, H) = (None, None)

        if W is None or H is None:
		          (H, W) = frame.shape[:2]
        # detecion
        #timer_start = time.time()
        # classes, scores, boxes = self.model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.frame_x, self.frame_y), swapRB=True, crop=False)
    	self.net.setInput(blob)
    	timer_start = time.time()
    	layerOutputs = self.net.forward(self.ln)
        timer_end = time.time()

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
    		for detection in output:
    			scores = detection[5:]
    			classID = np.argmax(scores)
    			confidence = scores[classID]
    			if confidence > CONFIDENCE_THRESHOLD:
    				box = detection[0:4] * np.array([W, H, W, H])
    				(centerX, centerY, width, height) = box.astype("int")
    				x = int(centerX - (width / 2))
    				y = int(centerY - (height / 2))
    				boxes.append([x, y, int(width), int(height)])
    				confidences.append(float(confidence))
    				classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    	if len(idxs) > 0:
    		for i in idxs.flatten():
    			(x, y) = (boxes[i][0], boxes[i][1])
    			(w, h) = (boxes[i][2], boxes[i][3])
    			color = (255,255,0)#[int(c) for c in COLORS[classIDs[i]]]
    			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                print(i)
                print(classIDs)
                print(confidences)
                print(classIDs[i])
                print(confidences[i])
                text = "something" #{}: {:.4f}".format(CLASS_NAMES[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # # drawing label, boxes etc.
        # for (class_id, score, box) in zip(classes, scores, boxes):
        #     color = COLORS[int(class_id) % len(COLORS)]
        #     label = "%s : %f" % (CLASS_NAMES[class_id], score)
        #
        #     cv2.rectangle(frame, box, color, 2)
        #     cv2.putText(frame, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        #
        # fps_label = "FPS : %.2f" % (1 / (timer_end - timer_start))
        # cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # convert back to IMG_MSG
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        except CvBridgeError as e:
            print(e)

        self.img_detected_pub.publish(img_msg)

def read_class_name():
    #long path for using with rosrun
    with open("src/object_recognition3D/src/yolo_utils/coco.names", "r") as f:
        CLASS_NAMES = [class_name.strip() for class_name in f.readlines()]
    COLORS = np.random.randint(0, 255, size=(len(CLASS_NAMES), 3), dtype="uint8")
    print(CLASS_NAMES)
    print(COLORS)
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
