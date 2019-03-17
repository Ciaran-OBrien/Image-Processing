
# Starter Pack for Image Processing

# (C) Dr Jane Courtney 2018

# import the necessary packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

######### Reading Images #########

# Opening an image from a file:
# I = cv2.imread("image.jpg")

# Opening an image using a File Open dialog:
f = easygui.fileopenbox()
I = cv2.imread(f)

# Capturing an image from a webcam:
# Video = cv2.VideoCapture(0)
# (check, I) = Video.read()

# Video capture from a file:
# Video = cv2.VideoCapture("Zorro.mp4")
# (check, I) = Video.read()

# while check:
	# cv2.imshow("image", I)
	
	# key = cv2.waitKey(1)

	# # if the 'q' key is pressed, quit:
	# if key == ord("q"):
		# break
	
	# # Next Frame:
	# (check, I) = Video.read()

# Video.release()

######### Writing Images #########

# Writing an image:
# cv2.imwrite("image.jpg",I)

######### Showing Images #########

# Showing an image on the screen (OpenCV):
cv2.imshow("image", I)
key = cv2.waitKey(0)

# Showing an image on the screen (MatPlotLib):
# I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
# plt.imshow(I) 
# plt.show() 

######### Colourspaces #########

# Converting to different colour spaces:
# RGB = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
# HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
# YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
# G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# Showing each image on the screen in a different window (OpenCV):
# cv2.imshow("Original", I)
# cv2.imshow("HSV", HSV)
# cv2.imshow("YUV", YUV)
# cv2.imshow("Grayscale", G)
# key = cv2.waitKey(0)

######### Pixels #########

# # Accessing a pixel's value:
# B = I[200,300,0]
# BGR = I[200,300]
# print "The blue value @ (200,300) is ", B
# print "The pixel value @ (200,300) is ", BGR

# Setting a pixel's value:
# I[200,300,0] = 255
# I[205,300] = (255,0,0)
# cv2.imwrite("image.bmp",I)

# Using the colon operator:
# I[190:210,190:210] = (0,255,0)
# Pixel = I[200,200,:]
# print "The pixel value @ (200,200) is ", Pixel

######### Drawing #########

# Keeping a copy:
# Original = I.copy() 

# # Drawing a line:
# cv2.line(img = I, pt1 = (200,200), pt2 = (500,600), color = (255,255,255), thickness = 5) 

# # Drawing a circle:
# cv2.circle(img = I, center = (800,400), radius = 50, color = (0,0,255), thickness = -1)

# # Drawing a rectangle:
# cv2.rectangle(img = I, pt1 = (500,100), pt2 = (800,300), color = (255,0,255), thickness = 10)

# Getting the size of the image:
# size = np.shape(I)

######### User Input #########

# Capturing user input:
# def draw(event,x,y,flags,param): 
	# if event == cv2.EVENT_LBUTTONDOWN: 
		# cv2.circle(img = I, center = (x,y),radius = 5, color = (255,255,255), thickness = -1) 
		# cv2.imshow("image", I) 
			
# cv2.namedWindow("image") 
# cv2.setMouseCallback("image", draw) 
# cv2.imshow("image", I)
# key = cv2.waitKey(0)

# A handy way to use the waitkey....

# while True:
	# cv2.imshow("image", I)
	# key = cv2.waitKey(0)

	# # if the 'r' key is pressed, reset the image:
	# if key == ord("r"):
		# I = Original.copy()

	# # if the 'q' key is pressed, quit:
	# elif key == ord("q"):
		# break
