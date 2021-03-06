# Target detection code for dot target
# Created by Nick Zhang

# This file includes standard routines used in Gatech DBF MedExpress vision team
# it should be the start point for your code if you decide to implement different target pipelines
 
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
import collections

from timeUtil import execution_timer
t = execution_timer(True)

#helper functions/routines

#draw lines on img
def draw_lines(img, lines, color=[255, 0, 0], thickness=1):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def show(img):
    plt.imshow(img)
    plt.show()

# show gray image
def showg(img):
    plt.imshow(img,cmap='gray')
    plt.show()
# show heat image
def showh(img):
    plt.imshow(img,cmap='gist_heat')
    plt.show()

# find area of interest in an image
# IMAGE:    original image
# RETURN:   a list of coordinates defining boxes that contain area of interest
# This should be implemented after we find a cheap way to eliminate some backgrounds
# Your findTarget should be able to take AOI as an input, and examine only the AOIs.
# For now you may ignore this

# TODO - move what's in pre() here
# image -> original RGB image
# return: a labeled image as aoi and label count
aoiType = collections.namedtuple('aoiType', ['labels', 'num_features'])

def distance(p1,p2):
    dx = p1[0]-p2[0]
    dy = p1[1]-p2[1]
    return (dx**2+dy**2)**0.5

# main procedure to find target in image
# for now it should draw visible marker around target
# Current AOI format: non-AOI is zero, each section of AOI is labeled with a non-zero identifier
# see findAOI() for definition
# TODO - test how much performance increase is gained from aoi
def findTarget(image, aoi = None):
    #if aoi is None:
    #    aoi = aoiType(np.ones(image.shape[0],image.shape[1]),1)

    t.s('cvtC+Caussian')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 9
    gauss_gray_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    t.e('cvtC+Caussian')

    #find edge & contour
    # if we do hist balance
    #low_threshold = 800
    #high_threshold = 1200
    t.s('canny')
    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(gauss_gray_image, low_threshold, high_threshold)
    t.e('canny')

    t.s('fd cont')
    im2, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_TC89_L1)
    t.e('fd cont')

    # XXX- if countours = NOne
    # TODO - can we do this vector style?
    target_count = 0
    for c in contours:
        t.s('arclen')
        peri = cv2.arcLength(c, True)
        convex = cv2.convexHull(c)
        conv_peri = cv2.arcLength(convex, True)
        t.e('arclen')

        # continue if it's not really a convex shape
        if (peri>conv_peri*1.1):
            continue
        
        t.s('minAreaRect')
        box = cv2.minAreaRect(c)
        points = cv2.boxPoints(box)
        t.e('minAreaRect')

        _, (w,h), _ = box
        # remove noise and prevent div by zero
        if (w<5) or (h<5):
            continue
        # we know the longer edge would be at least 15 pixels, at current configuration
        if max(w,h)<12:
            continue

        # restrict aspect ratio
        aspectRatio = max(w,h)/min(w,h)

        #compute the solidity of the original contour
        area = cv2.contourArea(c)
        hullArea = cv2.contourArea(convex)
        solidity = area / float(hullArea)

        if  (solidity < 0.9 or aspectRatio>1.5 or aspectRatio < 1.1):
            continue

        contourIdx = -1;
        thickness = 2
        color = [0,0,255]
        box2draw = np.int0([points])

        t.s('validation')
        # final check
        centerx = np.average(box2draw[0,:,0])
        centery = np.average(box2draw[0,:,1])
        d1 = distance(box2draw[0,0],box2draw[0,2])
        d2 = distance(box2draw[0,1],box2draw[0,3])
        diagonal = np.average([d1,d2])
        radius = diagonal*0.35/2

        roi = np.zeros([image.shape[0],image.shape[1]], dtype=np.uint8)
        cv2.circle(roi,(int(centerx),int(centery)),int(radius),1,-1)
        color_center = cv2.mean(gray_image,roi)[0]

        roi = np.zeros([image.shape[0],image.shape[1]], dtype=np.uint8)
        cv2.drawContours(roi, box2draw, contourIdx, 1, -1)
        cv2.circle(roi,(int(centerx),int(centery)),int(radius),-1,-1)
        color_edge = cv2.mean(gray_image,roi)[0]

        t.track('color center',color_center)
        t.track('color edge',color_edge)
        t.e('validation')
        if not (color_center+20<color_edge):
            t.track('bad edge',color_edge)
            t.track('bad center',color_center)
            continue

        cv2.drawContours(image, box2draw, contourIdx, color, thickness)
        cv2.circle(image,(int(centerx),int(centery)),3*int(radius),[255,0,0],5)
        target_count +=1

    # only tracks if we actually find at least one target
    if target_count>=1:
        t.track('target count',target_count)

        

    return image


#test function
def box_around_labels(img, labels,num_features):
    # Iterate through all labels 
    final_features = num_features
    for i in range(1, num_features+1):
        # Find pixels with each label value
        nonzero = (labels == i).nonzero()

        # return if this feature is removed by previous iterations
        if (len(nonzero[0])==0):
            final_features = final_features-1
            return img
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # remove this aoi from labels so we don't do unnecessary calculation
        # this increase speed by 200%
        cv2.rectangle(labels, bbox[0], bbox[1],0, -1)
        
    t.track('num_features',final_features)
    # Return the image
    return img

# expand a grayscale image to a color image
def expand_grayimg(image):

    _image = np.zeros([image.shape[0],image.shape[1],3], dtype=np.uint8)
    _image[:,:,0] = image
    _image[:,:,1] = image
    _image[:,:,2] = image
    image = _image
    return image

#drop some frames to speed things up
counter = 0;
last_detection = None;
avg_item = 0;
avg_value = None;

# this is the direct callback handler of a new image/frame
# it decides whether to drop the frame or process it based on performance
# this will be determined by loop frequence in real application
# for now it just drop a fixed ratio of frames
def pipeline(image):
    global last_detection
    global counter
    global avg_item
    global avg_value
    counter = counter + 1
    if True |(counter%2 == 0) | (last_detection is None):
        loop_start_time = time.time()
        
        t.s()
        processed = findTarget(image)
        t.e()

        duration = time.time() - loop_start_time
        if (avg_value is None):
            avg_item = avg_item + 1
            avg_value = duration
        else:
            avg_value = avg_item * avg_value + duration
            avg_item = avg_item + 1
            avg_value = avg_value / avg_item

        last_detection = processed
        return last_detection
    else:
        return last_detection

# debug routine for an image 
def test_image(filename):
    test_image = cv2.imread(filename)

    image = findTarget(test_image)
    edges = image
    _edges = np.zeros([edges.shape[0],edges.shape[1],3], dtype=np.uint8)
    _edges[:,:,0] = edges
    _edges[:,:,1] = edges
    _edges[:,:,2] = edges
    edges = _edges
    image = edges
    
    plt.imshow(image)
    plt.show()
    return
        

# test image filename
filename = "/Users/Nickzhang/uav_challenge/test_module/resources/target/high1.png"
filename = "/Users/Nickzhang/uav_challenge/test_module/resources/hard/hard1.png"
#filename = "/Users/Nickzhang/uav_challenge/test_module/resources/target/high3.png"
#test_image(filename)
#exit()
output_filename = "/Users/Nickzhang/uav_challenge/test_module/resources/output/output.mp4"
clip = VideoFileClip("/Users/Nickzhang/uav_challenge/test_module/resources/DJI_0001.MOV")
processed_clip = clip.fl_image(pipeline) #NOTE: this function expects color images!!
processed_clip.write_videofile(output_filename, audio=False)
print("average processing frequency = " + str(1/avg_value))
t.summary()
