import numpy as np
import cv2
class Thresholder():
	def __init__(self):
		pass

	def sat_value(self, lane):
		hsv = cv2.cvtColor(lane, cv2.COLOR_BGR2HSV)
		hls = cv2.cvtColor(lane, cv2.COLOR_BGR2HLS)

		combined = np.zeros(lane.shape[:2])
		s = hls[:,:,-1]
		v = hsv[:,:,-1]
		combined[((s >= 70 )) & ((v >= 150))] = 1

		return combined

	def binary_thresh(self, g, th=190, t1=50, t2=255):
		img = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
		_, mask = cv2.threshold(img, thresh=th, maxval=255, type=cv2.THRESH_BINARY)
		grayb = np.zeros_like(img) 

		grayb[(mask > t1) & (mask <= t2)] = 1
		return grayb

	def white_mask(self, lane, sensitivity = 50):
		hsv = cv2.cvtColor(lane, cv2.COLOR_BGR2HSV)

		# define range of white color in HSV
		# change it according to your need !
		hMin = 0;sMin = 0;vMin = 180
		hMax = 179;sMax = 29;vMax = 255
		lower_white = np.array([hMin, sMin, vMin])
		upper_white = np.array([hMax, sMax, vMax])

		# Threshold the HSV image to get only white colors
		mask = cv2.inRange(hsv, lower_white, upper_white)
		# Bitwise-AND mask and original image
		res = cv2.bitwise_and(lane,lane, mask= mask)
		res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
		return res

	def combine(self, thresh1, thresh2):
		combined_binary = np.zeros_like(thresh1)
		combined_binary[(thresh1==1) | (thresh2==1)] = 1
		return combined_binary

	def get_thresh(self, img):

		hls_dark = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
		hsv_dark = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		#for left lane
		satval = np.zeros(img.shape[:2])
		s = hls_dark[:,:,-1]
		v = hsv_dark[:,:,-1]
		satval[((s >= 70 )) & ((v >= 150))] = 1

		#for right lane
		binary = self.binary_thresh(img)
		white  = self.white_mask(img)

		combined = np.zeros_like(binary)
		combined[(satval==1) | ((binary==1) | (white == 1))] = 1
		return combined
	    # return final_mask