import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


def main():
    master('big1.png', 4)


def master(file_name, worker_count):
    # Load the desired image
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(gray)
    # Split the image into chunks
    chunks = []
    img_h, img_w = img.shape
    chunk_count = int(math.sqrt(worker_count))
    chunk_w = img_w // chunk_count
    chunk_h = img_h // chunk_count
    # test = img[:chunk_w, :chunk_h]
    # for i in test:
    #     for j in i:
    #         print(j, end='')
    #     print('')
    for i in range(chunk_count):
        for j in range(chunk_count):
            chunks.append(img[chunk_w * i : chunk_w * (i+1), chunk_h * j : chunk_h * (j+1)])
    # for id, chunk in enumerate(chunks):
    #     cv2.imshow(str(id), chunk)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    futures = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for index, chunk in enumerate(chunks):
            futures.append(executor.submit(work, chunk, index, worker_count))

        for future in futures:
            result = future.result()
            plot(img, result)
    cv2.imwrite('linesDetected_' + file_name, img)


def work(image, index, count):
    return hough_transform(image, index, count)


def plot(image, list):
    for dict in list:
        r = dict['r']
        theta = math.pi * dict['theta'] / 180
        # Stores the value of cos(theta) in a
        cos = np.cos(theta)

        # Stores the value of sin(theta) in sin
        sin = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = cos * r

        # y0 stores the value rsin(theta)
        y0 = sin * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-sin))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (cos))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-sin))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (cos))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)


def hough_transform(image, worker_index, worker_count, peak_vicinity=15, theta_precision=5, pixel_intensity_threshold=200):
    img_h, img_w = image.shape
    diag = int(math.sqrt(img_w ** 2 + img_h ** 2))

    # The precision of the search is of 1 degree and of one pixel
    accumulator = np.zeros((diag, 180))

    # Mark down in the accumulator the lines that can go through each point from the image
    for i, line in enumerate(image):
        for j, val in enumerate(line):
            if val > pixel_intensity_threshold:
                for theta in range(0, 180, theta_precision):
                    # x*cos(theta) + y*sin(theta) = r
                    r = int(j * math.cos(math.pi * theta / 180) + i * math.sin(math.pi * theta / 180))
                    accumulator[r][theta] = accumulator[r][theta] + 1

    # Find out the biggest frequency in the accumulator
    max = 0
    for i in range(diag):
        for j in range(180):
            if accumulator[i][j] > max:
                max = accumulator[i][j]

    lines = []
    # y = (r - x * cos(theta)) / sin(theta)
    for i in range(diag):
        for j in range(180):
            if max - accumulator[i][j] <= peak_vicinity:
                lines.append({"theta": j, "r": i + diag * (worker_index // math.sqrt(worker_count))})
    return lines


if __name__ == '__main__':
    main()
