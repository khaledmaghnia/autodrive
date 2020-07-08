import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Calibrator():
	def __init__(self, loc, nx, ny):
		self.chessboards = glob.glob(loc+'/*.jpg')
		self.nx = nx
		self.ny = ny

	def calibrate(self, verbose=False, save=True):
		if verbose:
			print('Initializing calibration....')
		img_points = []
		obj_points = []

		objp = np.zeros((self.nx*self.ny, 3), dtype=np.float32)
		objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)

		if verbose:
			print('Processing camera matrices...\n')

		for chessboard in self.chessboards:
			img = cv2.imread(chessboard, 0)
			ret, corners = cv2.findChessboardCorners(img, (self.nx, self.ny), None)
			if ret:
				img_points.append(corners)
				obj_points.append(objp)

		if verbose:
			c_image = cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
			plt.imshow(c_image, cmap='gray')
			plt.show()

		ret, self.cmtx, self.dmtx, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape, None, None)
		if ret:
			print('Camera calibrated successfully. Camera matrices are stored as "cmtx" and "dmtx"')
		else:
			print("Calibration failed. Make sure you have correctly included the location containing chess-board images.")


		if save:
			self.save_camera_matrices()


	def load_camera_matrices(self, cmtx, dmtx):
		self.cmtx = np.load(cmtx)
		self.dmtx = np.load(dmtx)

	def undistort(self, image):
		try:
			return cv2.undistort(image, self.cmtx, self.dmtx, None, self.cmtx)
		except e:
			print('Please load the camera matrices first or calibrate the camera.\
				\nUsage: calibrator.calibrate()\tcalibrator.load_camera_matrices(cmtx_loc, dmtx_loc)')

	def save_camera_matrices(self):
		os.makedirs("files", exist_ok=True)

		np.save('files/cmtx', self.cmtx)
		np.save('files/dmtx', self.dmtx)
		print(f'Saved camera matrices as: "cmtx.npy" & "dmtx.npy" in {os.getcwd()+"/files"}')