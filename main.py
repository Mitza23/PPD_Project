# Python program to illustrate HoughLine
# method for line detection
import math

import cv2
import numpy
import numpy as np


def hough_transform(file_name, peak_vicinity=1, theta_precision=5, pixel_intensity_threshold=200):
    # Load the desired image
    img = cv2.imread(file_name)

    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image = np.asarray(gray)  # (height, width)
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
                lines.append({"theta": j, "r": i})
                r = i
                theta = math.pi * j / 180
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
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imwrite('linesDetected_' + file_name, img)
    # cv2.imshow("linesDetected", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for i in range(diag):
        for j in range(180):
            print(int(accumulator[i][j]), end=' ')
        print('')

    # Normalize values in the Hough space for better visibility
    for i in range(diag):
        for j in range(180):
            accumulator[i][j] = (255 // max) * accumulator[i][j]

    cv2.imwrite('houghSpace_' + file_name, accumulator)
    # cv2.imshow("HoughSpace", accumulator)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return lines


hough_transform("6.png", peak_vicinity=2)
