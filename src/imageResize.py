import cv2

import numpy as np

def resizeToMNISTFormat(imageFile):
    image = cv2.imread(imageFile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.equalizeHist(image)
    res = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_NEAREST)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if res[i, j] < 100:
                res[i, j] = 0.
    return res
