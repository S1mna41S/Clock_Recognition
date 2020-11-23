from os import path
import cv2 as cv
import numpy as np
import math

PICTURES_FOLD = 'Pictures'

image = cv.imread(path.normpath(path.join(PICTURES_FOLD, 'probe_pic1.jpg')))
image_size = image.shape[:2]

imgCanny = cv.Canny(image, 200, 200)
cdst = cv.cvtColor(imgCanny, cv.COLOR_GRAY2BGR)

lines = cv.HoughLines(imgCanny, 1, np.pi/180, 150, None, 0, 0)


cv.imshow("Image", image)
cv.imshow("Canny image", imgCanny)
cv.waitKey(0)
cv.destroyAllWindows()
