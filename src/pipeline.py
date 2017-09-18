# Created by Nick Zhang

# This is a pipeline that tracks 2 by 2 chessboard pattern in an image/video

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import pickle
import warnings
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

#matplotlib.style.use('ggplot')

#helper functions

#draw lines on img
def draw_lines(img, lines, color=[255, 0, 0], thickness=1):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def show(img):
    plt.imshow(img)
    plt.show()

def showg(img):
    plt.imshow(img,cmap='gray')
    plt.show()

# find area of interest in an image
# @return a list of coordinates defining boxes that contain area of interest
def findAOI(image):

# do a histogram, find most common pixels

# label all special pixels

# draw box around special pixels

    return [image.shape]

def findCross(image, aoi):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    gray_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    #normalize

    low_threshold = 200
    high_threshold = 300
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    return edges

    plt.imshow(edges,cmap='gray')
    plt.show()
    #XXX - min_line, max_line need to be dynamically calculated form size of AOI
    rho = 1
    theta = np.pi/180*3
    threshold = 7
    min_line_len = 5
    max_line_gap = 30
    
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        draw_lines(line_img, lines)
    
    w1 = 0.5
    w2 = 0.5

    #cv2.addWeighted(image, w1, line_img, w2, 0)
    _edges = np.zeros([edges.shape[0],edges.shape[1],3], dtype=np.uint8)
    _edges[:,:,0] = edges
    _edges[:,:,1] = edges
    _edges[:,:,2] = edges
    edges = _edges
    return cv3.addWeighted(edges, w1, line_img, w2, 0)

#XXX - reference only
def box_around_labels(labels,num_features):
    # Iterate through all labels 
    for i in range(1, num_features+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def expand_grayimg(image):

    _image = np.zeros([image.shape[0],image.shape[1],3], dtype=np.uint8)
    _image[:,:,0] = image
    _image[:,:,1] = image
    _image[:,:,2] = image
    image = _image
    return image

#some testing procedure for pre-processing
# TODO - break the function up to pre-process, ROI, etc.
def pre(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 17
    gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    #reduce color
    gray = gray//10*10
    #reduce scale, calculate background
    thumbnail = cv2.resize(gray, (40,20), interpolation = cv2.INTER_NEAREST)
    thumbnail = thumbnail.ravel()
    value, count = np.unique(thumbnail, return_counts = True)
    sort_index = np.argsort(count)
    # the most frequent 6 colors are considered mode
    mode = sort_index[-6:]
    mode = value[mode]
    #one for all non-mode pixels
    mask = np.isin(gray, mode, invert=True).astype(np.uint8)
    mask = mask*255
    kernel_size = 10
    #mask = cv2.GaussianBlur(mask,  (kernel_size, kernel_size), 10)
    mask = cv2.blur(mask, (kernel_size, kernel_size))
    threshold = 100
    mask[mask<=threshold] = 0

    # this variable defines pattern used to determine connectivity of features(none-zero pixles)
    # see generate_binary_structure and binary_dilation for extension
    connection = None
    #label all remaining non-zero pixels, these SECTIONS are candidates for ROI
    labels, num_features = label(mask, structure=connection)

    #eliminate sections with too few 'hot' pixels, this should filter out random dots.

    # initialization done for performance improvement
    # there is more than enough space since some labels will be discarded
    roi = np.zeros([num_features, 2, 2])
    roi_count = 0
    for i in range(1,num_features+1):
        # coordinates for all pixels
        coordinates = np.array((labels==i).astype(np.uint8).nonzero())
        num_pixels = coordinates.shape[1]
        
        #draw a box around the hot pixels
        # Identify x and y values of those pixels
        nonzeroy = np.array(coordinates[0])
        nonzerox = np.array(coordinates[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_size = (bbox[1][1]-bbox[0][1])*(bbox[1][0]-bbox[0][0])
        occupance = num_pixels / bbox_size
        # filter out sparce & tiny sections
        if (occupance > 0.6) and (bbox_size > 150) :
            roi[roi_count] = bbox
            roi_count = roi_count + 1
        if (roi_count > 0):
            # --- DEBUG
            # Draw the box on the image
            cv2.rectangle(image, bbox[0], bbox[1], (0,0,255), 4)

    ##### ---- DEBUG
    #print(len(value))
    #plt.figure();
    #plt.bar( np.arange(len(value) ), count, align='center' )
    #plt.show()
    return image

#drop some frames to speed things up
counter = 0;
last_detection = None;
avg_item = 0;
avg_value = None;
def pipeline(image):
    global last_detection
    global counter
    global avg_item
    global avg_value
    counter = counter + 1
    if (counter%2 == 0) | (last_detection is None):
        t = time.time()
        edges = pre(image)
        duration = time.time() - t
        if (avg_value is None):
            avg_item = avg_item + 1
            avg_value = duration
        else:
            avg_value = avg_item * avg_value + duration
            avg_item = avg_item + 1
            avg_value = avg_value / avg_item

        last_detection = edges
        return last_detection
    else:
        return last_detection

    

# test image filename
filename = "/Users/Nickzhang/uav_challenge/test_module/resources/target/high1.png"
filename = "/Users/Nickzhang/uav_challenge/test_module/resources/target/full1.png"
#filename = "/Users/Nickzhang/uav_challenge/test_module/resources/target/high3.png"
test_image = cv2.imread(filename)

#image = pre(test_image)
#edges = image
#_edges = np.zeros([edges.shape[0],edges.shape[1],3], dtype=np.uint8)
#_edges[:,:,0] = edges
#_edges[:,:,1] = edges
#_edges[:,:,2] = edges
#edges = _edges
#image = edges
#
#plt.imshow(image)
#plt.show()
#exit()

output_filename = "/Users/Nickzhang/uav_challenge/test_module/resources/output/output.mp4"
clip = VideoFileClip("/Users/Nickzhang/uav_challenge/test_module/resources/GOPR0010.MP4").subclip(60,80)
processed_clip = clip.fl_image(pipeline) #NOTE: this function expects color images!!
processed_clip.write_videofile(output_filename, audio=False)
print("average processing frequency = " + str(1/avg_value))
