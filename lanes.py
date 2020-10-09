import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters

    #get the height
    #rint(image.shape)
    #704

    y1 = image.shape[0]
    y2 = int(y1*(3/5)) 
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):

    #left line on the img
    left_fit = []

    #right line on the img
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        #y = mx + b
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        
        #test if it works
        #print parameters
        #test passed, prints slope
        
        slope = parameters[0]
        intercept = parameters[1]
        
        #lines on the left with have a negative slope
        #as y declines / decreases

        #lines on the right will have a positive slope
        #as x grows, so does y

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    
    #test 
    #rint(left_fit)
    #rint(right_fit)
    #passed

    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    
    #test
    #rint(left_fit_average, 'left')
    #rint(right_fit_average, 'right')
    #passed

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])

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
        for x1, y1, x2, y2 in lines:
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


#canny_image = canny(frame)
#cropped_image = region_of_interest(canny_image)

#this is the algorithm, it is really powerful
#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

#averaged_lines = average_slope_intercept(frame, lines)

#line_image = display_lines(frame, averaged_lines)

#our_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

#name our window
#cv2.imshow('AI EYES', our_image)

#display our image
#cv2.waitKey(0)



cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)

    #this is the algorithm, it is really powerful
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    averaged_lines = average_slope_intercept(frame, lines)

    line_image = display_lines(frame, averaged_lines)

    our_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    #name our window
    cv2.imshow('AI EYES', our_image)

    #display our image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




#for finding the lane
#assume right lane is THE RIGHT lane 
#we ignore UK, will work on this later

#we need to get the coordinates so I will use matplotlib to plot them
#p1 will be f(200, 700)
#p2 will be f(1100, 700)
#p3 will be f(500, 200)
#this marks a triangle

#we will need hough space for lane lines