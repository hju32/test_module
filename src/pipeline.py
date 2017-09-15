# Created by Nick Zhang

# This is a pipeline that tracks 2 by 2 chessboard pattern in an image/video

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import warnings
from moviepy.editor import VideoFileClip

#helper functions

#draw lines on img
def draw_lines(img, lines, color=[255, 0, 0], thickness=1):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# find area of interest in an image
# @return a list of coordinates defining boxes that contain area of interest
def findAOI(image):

# do a histogram, find most common pixels

# label all special pixels

# draw box around special pixels

    return [image.shape]

def findCross(image, aoi):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 3
    gray_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    #normalize

    low_threshold = 200
    high_threshold = 300
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)

    #XXX - min_line, max_line need to be dynamically calculated form size of AOI
    rho = 1
    theta = np.pi/200
    threshold = 30
    min_line_len = 5
    max_line_gap = 30
    
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        draw_lines(line_img, lines)
    
    w1 = 0.5
    w2 = 0.5

    return cv2.addWeighted(image, w1, line_img, w2, 0)
    

def pipeline(image):
    return findCross(image,1)

# test image filename
filename = "/Users/Nickzhang/uav_challenge/test_module/resources/target/high1.png"
#filename = "/Users/Nickzhang/uav_challenge/test_module/resources/target/full1.png"
filename = "/Users/Nickzhang/uav_challenge/test_module/resources/target/high3.png"
test_image = cv2.imread(filename)


output_filename = "/Users/Nickzhang/uav_challenge/test_module/resources/output/output.mp4"
clip = VideoFileClip("/Users/Nickzhang/uav_challenge/test_module/resources/GOPR0010.MP4").subclip(60,3*60+48)
processed_clip = clip.fl_image(pipeline) #NOTE: this function expects color images!!
processed_clip.write_videofile(output_filename, audio=False)
