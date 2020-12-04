from os import path
import cv2 as cv
import numpy as np
import argparse

PICTURES_FOLD = 'Pictures'
DEBUG = True


def get_path_to_picture(picture_name):
    return path.normpath(path.join(PICTURES_FOLD, picture_name))


def read_image(picture_name):
    image = cv.imread(get_path_to_picture(picture_name))
    rows, cols = image.shape[:2]
    if rows != cols:
        diff = abs(rows - cols)
        if rows > cols:
            image = image[diff // 2:-diff // 2, 0:cols]
            rows -= diff
        else:
            image = image[0:rows, diff // 2:-diff // 2]
            cols -= diff

    max_size = 720
    if rows > max_size:
        image = cv.resize(image, (max_size, max_size))
        rows = max_size
        cols = max_size
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image, img, rows, cols


def varp_polar(img):
    value = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    polar_image = cv.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), value, cv.WARP_FILL_OUTLIERS)
    return polar_image.astype(np.uint8)


def binary_inverse(image):
    _, thresh1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    return thresh1


def _neighbors_are_empty(image, rows, j, i):
    if i == rows - 1:
        return True
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
    current_len, longest_y = 0, 0
    for j, row in enumerate(image):
        for i, pixel in enumerate(row):
            if pixel and _neighbors_are_empty(image, rows, j, i):
                current_len = i
                longest_y = j
                break
        if DEBUG:
            print(f'{j}/{rows} - {current_len}')
        mean_height_1 += current_len
        list_heights[j] = current_len
        for i1, _ in enumerate(reversed(mask[j][:current_len + 1])):
            mask[j, i1] = 255

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
    if flag:
        intervals[(i, rows)] = area

    return mask, intervals


def what_time_is_this_interval(interval, rows):
    first, second = interval[0]
    mean = (first + second) / 2
    time = (((mean / rows) * 12) + 3) % 12
    return time


def what_time_is_it(first, second):
    first_rel = first / 12
    second_rel = second / 12
    first_cuted = abs(first - int(first))
    second_cuted = abs(second - int(second))
    first_is_hour = second_rel - first_cuted
    second_is_hour = first_rel - second_cuted
    if abs(second_is_hour) < abs(first_is_hour):
        print(f'It\'s {int(second)} chours {round(first_rel * 60)} minutes')
    else:
        print(f'It\'s {int(first)} chours {round(second_rel * 60)} minutes')


if __name__ == "__main__":
    if not DEBUG:
        print('Читаю картинку...')
    color_image, image, rows, cols = read_image(picture_name='VIaYFnij6yY.jpg')

    if not DEBUG:
        print('Разворачиваю...')
    polar_image = varp_polar(image)
    if DEBUG:
        cv.imshow("Polar Image", polar_image)

    if not DEBUG:
        print('Произвожу бинаризацию...')
    wb_image = binary_inverse(polar_image)
    if DEBUG:
        cv.imshow("W/B", wb_image)

    if not DEBUG:
        print('Ищу стрелки...')
    mask, intervals = find_arrows(wb_image, rows)

    cv.imshow("Image", color_image)

    if DEBUG:
        cv.imshow("Arrows", mask)

    if DEBUG:
        print(intervals)
    intervals_list = list(intervals.items())
    intervals_list.sort(key=lambda i: -i[1])
    if len(intervals_list) >= 2:
        intervals_list_cut = intervals_list[:2]
    else:
        intervals_list_cut = intervals_list
    if DEBUG:
        print(intervals_list_cut)
    time = []
    for interval in intervals_list_cut:
        time.append(what_time_is_this_interval(interval, rows))
    if DEBUG:
        print(time)

    what_time_is_it(time[0], time[1])

    cv.waitKey(0)
    cv.destroyAllWindows()
