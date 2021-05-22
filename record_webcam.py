import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np


flags.DEFINE_string('output', './video/test1.avi', 'path to output video')


def main(_argv):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, 5)
    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        return_value, frame = cap.read()
        if return_value == False:
            print('Video has ended or failed, try a different video format!')
            break

        cv2.imshow("result", frame)
        
        if FLAGS.output:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    
    cap.release()
    out.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass