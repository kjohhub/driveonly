#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import os
import sys
import time
import numpy as np
import tensorflow as tf
import cv2

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

if len(sys.argv) < 3:
    sys.exit()

src = sys.argv[1]
dst = sys.argv[2]
cap = cv2.VideoCapture(src)

# create directory if not exist
if os.path.isdir(dst) == False:
    os.mkdir(dst)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.compat.v1.Session(graph=detection_graph, config=config) as sess:
    index = 0
    while True:
      ret, image = cap.read()
      if ret == 0:
          break



      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      start_time = time.time()
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      elapsed_time = time.time() - start_time
      print('inference time cost: {}'.format(elapsed_time))
      #print(boxes.shape, boxes)
      #print(scores.shape,scores)
      #print(classes.shape,classes)
      #print(num_detections)

      # crop
      boxes = np.squeeze(boxes)
      if len(boxes) > 0:
        h, w, c = image.shape
        y_min = int(boxes[0, 0] * h)
        x_min = int(boxes[0, 1] * w)
        y_max = int(boxes[0, 2] * h)
        x_max = int(boxes[0, 3] * w)
        crop_image = image[y_min:y_max, x_min:x_max].copy()

        fname = "{0}{1:06d}.jpg".format(dst, index)
        cv2.imwrite(fname, crop_image)
        index += 1

    cap.release()
