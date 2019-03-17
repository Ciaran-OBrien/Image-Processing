import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui
import pyautogui, sys

I = image.imread("image.jpg")
clone = I.copy()
height, width, channels = I.shape

def draw(event, x,y ,flags, param):
	ref_point = [(x,y)]
	if event == cv2.EVENT_LBUTTONDOWN:
		ref_point.append((x+101,y-101))
		print(x,y)
		cv2.rectangle(I, ref_point[0],ref_point[1],(255,0,255),10)
		cv2.imshow("image",I)
		crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]] 
			
		#cv2.imshow("crop_img", crop_img) 



while True:
	cv2.imshow("image", I)
	cv2.namedWindow('image')
	cv2.setMouseCallback('image',draw)
	key = cv2.waitKey(0)
	
	# if the 'r' key is pressed, reset the image:
	if key == ord("r"):
		I = Original.copy()
	# if the 'q' key is pressed, quit:
	elif key == ord("q"):
		print("main while")
		break

