import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import json
import os
import signal
import time

from models import Thresholder, LaneDetector, Calibrator
from helpers import birdseye
from applications.yolo import YOLO

class Pipeline():
	def __init__(self, views = ['default'], use_tiny=False, download_file=False, use_cuda = False):
		self.calibrator = Calibrator('camera_cal', 8, 6)
		self.calibrator.load_camera_matrices('files/cmtx.npy', 'files/dmtx.npy')
		# self.calibrator.calibrate()

		self.thresholder = Thresholder()
		self.views = views

		self.lane_detector = LaneDetector()

		self.object_detector = YOLO(use_tiny, download_file, use_cuda=use_cuda)

	def get_view(self, key):
		view_map = {
			'default':None,
			'init': self.frame,
			'thresh':self.thresh,
			'warped': self.warped,
		}
		return view_map.get(key, 'init')

	def detect_trafficlight(self, box):
		ax = 10
		ay = 10

		masked = np.zeros_like(self.frame)
		masked[(box[1] - ay):(box[1]+box[3]+ay), (box[0] - ay):(box[0]+box[2]+ay)] = 1

		final = cv2.addWeighted(self.frame, self.frame, mask=masked)
		self.frame=final

	def pipeline(self, lane_only):
		undistorted = self.calibrator.undistort(self.frame)
		self.thresh = self.thresholder.get_thresh(undistorted)
		self.warped = birdseye.transform(self.thresh)
		lane_fit_img = self.lane_detector.fit_lanes(self.warped, self.lane_detector.left_fit, self.lane_detector.right_fit, self.lane_detector.left_fitx, self.lane_detector.right_fitx, verbose=False)
		left_curv, right_curv = self.lane_detector.measure_curvature_pixels(self.lane_detector.left_fit, self.lane_detector.right_fit)
		self.lane_mask = birdseye.inverse_transform(lane_fit_img)
		self.lane_mask_final = cv2.addWeighted(self.frame, 0.8, self.lane_mask, 0.2, 0)

		if not lane_only:
			self.lane_mask_final = self.object_detector.draw_boxes(self.frame, self.lane_mask_final)

		final_image = self.extract_views()

		curr_time = time.time()
		diff = curr_time - self.start_time
		self.start_time = curr_time

		fps = int(1/diff)

		cv2.putText(final_image,f"Left curvature: {left_curv}", (15, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (125, 40, 32), 2)
		cv2.putText(final_image,f"Right curvature: {right_curv}", (15, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
		cv2.putText(final_image,f"FPS: {fps}", (final_image.shape[1]-150, 220), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 2)

		return final_image

	def run_video(self, input_video, output_video, lane_only = True):

		#video for performing detection
		cap = cv2.VideoCapture(input_video)

		#for using the fit from initial lane in others wihout performing sliding windows again and again
		if output_video is not None:
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			out = cv2.VideoWriter(f'outputs/{output_video}', fourcc, cap.get(cv2.CAP_PROP_FPS), (1120, 780))

		self.start_time = time.time()
		while True:
			ret, self.frame = cap.read()
			if ret:
				final_image = self.pipeline(lane_only)
				cv2.imshow('Lane', final_image)
				if cv2.waitKey(1) & 0xFF==ord('q'):
					break
				if output_video is not None:
					out.write(final_image)
			else:
				break
		cap.release()
		cv2.destroyAllWindows()

	def run_image(self):
		# self.frame = cv2.imread('test_images/img.jpg')
		self.frame = cv2.imread('test_images/road.jpg')
		self.start_time = time.time()
		final_image = self.pipeline(False)
		cv2.imshow("img", final_image)
		cv2.waitKey()
		cv2.destroyAllWindows()

	def extract_views(self):
		if len(self.views) > 4:
			print('Not more than 4 views are allowed. Reducing the number to 4.')

		view_images = map(self.get_view, self.views[:4])
		final_image = np.zeros((780, 1120, 3), dtype=np.uint8)
		for i, image in enumerate(view_images):
			# print(image.shape)
			try:
				image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
			except:
				pass
			image = cv2.resize(image, (280, 140))
			final_image[0:140, i*280:i*280+280, :] = np.uint8(image/np.max(image)*255)
		lane_mask_final = cv2.resize(self.lane_mask_final, (1120, 600))
		final_image[180:, :] = np.uint8(lane_mask_final)
		return final_image

if __name__=='__main__':

	input_video = 'test_videos/challenge_video.mp4'
	output_video = None

	ap = argparse.ArgumentParser()
	ap.add_argument('-t', action='store_true', required=False, help='Use tiny yolo model')
	ap.add_argument('--input', required=False, type=str, help='Input video for the autodrive model')
	ap.add_argument('--output', required=False, type=str, help='Name of output video to store the output of autodrive model')
	ap.add_argument('-g', required=False, type=str, help='Add gpu support to perform object detection')
	args = vars(ap.parse_args())

	#whether to use yolo tiny model or not
	use_tiny = args['t']
	use_cuda = args['gpu']

	if args['input'] is not None:
		input_video = args['input'].strip("'")
	if args['output'] is not None:
		output_video = args['output'].strip("'")

	#if weights present then don't download, otherwise download
	download_file = True
	if use_tiny:
		if os.path.isfile('files/yolov3-tiny.weights'):
			download_file = False
	else:
		if os.path.isfile('files/yolov3.weights'):
			download_file = False


	pipeline = Pipeline(views=['thresh', 'warped'], use_tiny=use_tiny, download_file=download_file, use_cuda=use_cuda)

	#run the pipeline on a single image
	# pipeline.run_image()
	
	#run the pipeline on specified video
	pipeline.run_video(input_video, output_video, lane_only=False)

	#kill the application
	os.kill(os.getpid(), signal.SIGTERM)