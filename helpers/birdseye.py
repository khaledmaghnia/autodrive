import numpy as np
import cv2

src = np.float32([
    [280, 720],
    [555, 480],
    [740, 480],
    [1120, 720]
])

dst = np.float32([
    [280, 720],
    [280, 0],
    [1100, 0],
    [1100, 720]
])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

def transform(image):
	return cv2.warpPerspective(image, M, image.shape[:2][::-1], flags=cv2.INTER_LINEAR)

def inverse_transform(image):
	return cv2.warpPerspective(image, Minv, image.shape[:2][::-1], flags=cv2.INTER_LINEAR)