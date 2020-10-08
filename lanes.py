import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    #cvt to grayscale which has 1 channel
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #we need to remove the image noise now by using blur / gaussianBlur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    #find edges (lane path) using canny

    #black shows almost no change in gradient INTENSITY troughout the image
    #i.e. small changes in britghtness are ignored and are just set to black
    low_threshold = 50

    #white lines show us change in gradient
    #they show EXCEEDING of the high_threshold
    high_threshold = 150

    canny = cv2.Canny(blur, low_threshold, high_threshold)

    return canny

def display_lines(image, lines):

    line_image = np.zeros_like(image)
    if lines is not None:

        #this will print all lines in a 2D array
        for line in lines:

            #we however want to reshape it in 1D
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0 ,0), 10)

    return line_image

def region_of_interest(image):

    #return the region of the triangle
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)

    #fill the mask
    cv2.fillPoly(mask, polygons, 255)

    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


#load image
image = cv2.imread('test_image.jpg')

#we use COPY to skip making many, rapid changes
#on the original image
lane_image = np.copy(image)

canny = canny(lane_image)
cropped_image = region_of_interest(canny)

lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
line_image = display_lines(lane_image, lines)

our_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

#name our window
cv2.imshow('AI EYES', our_image)

#display our image
cv2.waitKey(0)



#for finding the lane
#assume right lane is THE RIGHT lane 
#we ignore UK, will work on this later

#we need to get the coordinates so I will use matplotlib to plot them
#p1 will be f(200, 700)
#p2 will be f(1100, 700)
#p3 will be f(500, 200)
#this marks a triangle

#we will need hough space for lane lines