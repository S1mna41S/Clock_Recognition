from os import path
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

PICTURES_FOLD = 'Pictures'


def get_path_to_picture(picture_name):
    return path.normpath(path.join(PICTURES_FOLD, picture_name))


image = cv.imread(get_path_to_picture('probe_pic1.jpg'), cv.IMREAD_GRAYSCALE)
rows, cols = image.shape

img = image.astype(np.float32)

value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))

polar_image = cv.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv.WARP_FILL_OUTLIERS)


polar_image = polar_image.astype(np.uint8)
cv.imshow("Polar Image", polar_image)
cv.imshow("Image", image)

cv.waitKey(0)
cv.destroyAllWindows()
