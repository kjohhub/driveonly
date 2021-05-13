import cv2
import sys
import os

if len(sys.argv) < 3:
    sys.exit()

src = sys.argv[1]
dst = sys.argv[2]
cap = cv2.VideoCapture(src)
idx = 0

os.mkdir(dst)

while True:
    ret, image = cap.read()
    if ret == 0:
        break
    else:
        fname = "{0}{1:05d}.jpg".format(dst, idx)
        cv2.imwrite(fname, image)
        idx += 1

