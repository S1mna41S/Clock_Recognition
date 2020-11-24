from os import path
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

PICTURES_FOLD = 'Pictures'


def get_path_to_picture(picture_name):
    return path.normpath(path.join(PICTURES_FOLD, picture_name))


def read_image(picture_name):
    image = cv.imread(get_path_to_picture(picture_name))
    image = cv.resize(image, (500, 500))
    rows, cols = image.shape[:2]
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image, img, rows, cols


def varp_polar(img):
    value = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    polar_image = cv.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), value, cv.WARP_FILL_OUTLIERS)
    return polar_image.astype(np.uint8)


# def find_boundaries(img):
#     ret, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
#     # noise removal
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
#
#     # sure background area
#     sure_bg = cv.dilate(opening, kernel, iterations=3)
#
#     # Finding sure foreground area
#     dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
#     ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
#
#     # Finding unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv.subtract(sure_bg, sure_fg)
#
#     # Marker labelling
#     ret, markers = cv.connectedComponents(sure_fg)
#
#     # Add one to all labels so that sure background is not 0, but 1
#     markers = markers + 1
#
#     # Now, mark the region of unknown with zero
#     markers[unknown == 255] = 0
#
#     h, w = markers.shape
#     markers = cv.CreateMat(h, w, cv.CV_32SC1)
#     h, w = img.shape
#     img = cv.CreateMat(h, w, cv.CV_8UC3)
#
#     markers = cv.watershed(img, markers)
#     img[markers == -1] = [255, 0, 0]

# def find_boundaries(img, rows, cols):
#     ## medianBlur, threshold and morph-close-op
#     median = cv.medianBlur(img, ksize=17)
#     retval, threshed = cv.threshold(median, 110, 255, cv.THRESH_BINARY_INV)
#     closed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, np.ones(shape=(15, 15)))
#
#     ## Project to the axis
#     xx = np.sum(closed, axis=0) / rows
#     yy = np.sum(closed, axis=1) / cols
#
#     ## Threshold and find the nozero
#     xx[xx < 60] = 0
#     yy[yy < 100] = 0
#
#     ixx = xx.nonzero()
#     iyy = yy.nonzero()
#     x1, x2 = ixx[0][0], ixx[0][-1]
#     y1, y2 = iyy[0][0], iyy[0][-1]
#
#     ## label on the original image and save it.
#     res1 = cv.rectangle(img.copy(), (x1, y1), (x2, y2), (0, 0, 255), 2)
#     res2 = img[y1:y2, x1:x2]
#     cv.imwrite("result1.png", res1)
#     cv.imwrite("result2.png", res2)

def binary_inverse(image):
    _, thresh1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    return thresh1


def _neighbors_are_empty(image, rows, j, i):
    if 0 < j < rows - 1 and image[j - 1, i] and image[j - 1, i + 1] and image[j, i + 1] and image[j + 1, i] \
            and image[j + 1, i + 1]:
        return True
    elif j == 0 and image[j, i + 1] and image[j + 1, i] and image[j + 1, i + 1]:
        return True
    elif image[j - 1, i] and image[j - 1, i + 1] and image[j, i + 1]:
        return True
    else:
        return False


def _print_column(picture, column, color):
    for row in picture:
        row[column] = color


def find_arrows(image, rows):
    mask = np.zeros(image.shape, image.dtype)
    mean_height_1 = 0
    list_heights = [0] * rows
    for _ in range(image.shape[0]):
        max_len, longest_y = 0, 0
        for j, row in enumerate(image):
            for i, pixel in enumerate(row):
                if mask[j, i]:
                    break
                if pixel and _neighbors_are_empty(image, rows, j, i):
                    if i > max_len:
                        max_len = i
                        longest_y = j
                    break

        # print(f'{longest_y} - {max_len}')
        mean_height_1 += max_len
        list_heights[longest_y] = max_len
        for i, _ in enumerate(reversed(mask[longest_y][:max_len + 1])):
            mask[longest_y, i] = 255

    mean_height_1 = int(mean_height_1 / rows)
    _print_column(mask, mean_height_1, 120)

    mean_height_2, upper_mean_count = 0, 0
    for i in list_heights:
        if i > mean_height_1:
            mean_height_2 += i
            upper_mean_count += 1
    mean_height_2 = int(mean_height_2 / upper_mean_count)
    _print_column(mask, mean_height_2, 170)

    flag = False
    intervals = {}
    i = 0
    area = 0
    for row_num, row_hight in enumerate(list_heights):
        if row_hight > mean_height_2:
            if not flag:
                flag = True
                i = row_num
            area += row_hight - mean_height_2
        elif flag:
            flag = False
            intervals[(i, row_num)] = area
            area = 0


    return mask, intervals
    # arrow_area = 1
    # arrow_thick = 1
    # tr_image = np.transpose(image)
    # for i, row in enumerate(image[longest_y][:max_len]):
    #     x_current = max_len - i - 1
    #     for j, pixel in enumerate(tr_image[i]):


if __name__ == "__main__":
    color_image, image, rows, cols = read_image('4georgjensen_henning_koppel_black_30_ma.jpg')
    # find_boundaries(image, rows, cols)
    polar_image = varp_polar(image)
    wb_image = binary_inverse(polar_image)

    mask, intervals = find_arrows(wb_image, rows)

    cv.imshow("Image", color_image)
    # cv.imshow("Polar Image", polar_image)
    cv.imshow("W/B", wb_image)

    cv.imshow("Arrows", mask)
    print(intervals)

    cv.waitKey(0)
    cv.destroyAllWindows()
