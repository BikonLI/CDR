import cv2
import numpy as np
img = cv2.imread("frame_video/0/00000.jpg", cv2.IMREAD_REDUCED_COLOR_4)
backup = np.copy(img)

h, w, t = img.shape
for y in range(100, 2160, 15):
    for x in range(2400, 3840, 5):
        print(x, y)
        x1 = int(x / 3840 * w)
        y1 = int(y / 2160 * h)
        img = np.copy(backup)
        cv2.rectangle(img, (x1, y1), (x1 + 1, y1 + 1), color=(0, 0, 255), thickness=-1)
        cv2.imshow("name", img)
        cv2.waitKey(0)