# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:34:48 2023

@author: yul
"""

import cv2
import numpy as np

# Load the image
image = cv2.imread('C:/Users/ogray/OneDrive/Belgeler/dataset/single_prediction/1.jpg')



# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the color you want to search for
lower_color = np.array([0, 50, 50])
upper_color = np.array([10, 255, 255])

# Create a mask based on the color range
mask = cv2.inRange(hsv_image, lower_color, upper_color)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(mask, (5, 5), 0)

# Detect circles using HoughCircles function
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=100)

# If circles are detected, draw them on the image
if circles is not None:
    circles = np.round(circles[0, :]).astype(int)
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)

# Display the image with detected circles
cv2.imshow('Detected Circles', image)
cv2.waitKey(100000)
cv2.destroyAllWindows()