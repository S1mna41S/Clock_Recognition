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

sobel_horizontal = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
sobel_vertical = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)

cv.imshow("Image", image)
cv.imshow('Sobel horizontal', sobel_horizontal)
cv.imshow('Sobel vertical', sobel_vertical)

cv.waitKey(0)
cv.destroyAllWindows()
