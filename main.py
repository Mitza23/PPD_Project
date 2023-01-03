# Python program to illustrate HoughLine
# method for line detection
import math

import cv2
import numpy
import numpy as np

# Reading the required image in
# which operations are to be done.
# Make sure that the image is in the same
# directory in which this python program is
img = cv2.imread('1.png')

# Convert the img to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow('grayscale', gray)
# cv2.waitKey(0)
#
# # closing all open windows
# cv2.destroyAllWindows()

image = np.asarray(gray)  # (height, width)
img_h, img_w = image.shape
diag = int(math.sqrt(img_w**2 + img_h**2))
accumulator = np.zeros((diag, 180))
threshold = 200

for i, line in enumerate(image):
    for j, val in enumerate(line):
        if val > threshold:
            for theta in range(180):
                # x*cos(theta) + y*sin(theta) = r
                r = int(i * math.cos(math.pi*theta/180) + j * math.sin(math.pi*theta/180))
                accumulator[r][theta] = accumulator[r][theta] + 1



max = 0
for i in range(diag):
    for j in range(180):
        if accumulator[i][j] > max:
            max = accumulator[i][j]
    #     print(int(accumulator[i][j]), end=' ')
    # print('')

# y = r - x * cos(theta)
for i in range(diag):
    for j in range(180):
        if max - accumulator[i][j] < 1:
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
cv2.imwrite('linesDetected.jpg', img)
cv2.imshow("linesDetected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(diag):
    for j in range(180):
        accumulator[i][j] = (255 // max) * accumulator[i][j]
        print(int(accumulator[i][j]), end=' ')
    print('')

cv2.imwrite('houghSpace.jpg', accumulator)
cv2.imshow("HoughSpace", accumulator)
cv2.waitKey(0)
cv2.destroyAllWindows()
