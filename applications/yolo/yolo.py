import cv2
import torch

from applications.yolo.darknet import Darknet
from applications.yolo.utils import *
from helpers.download_weights import download

class YOLO():
	def __init__(self, use_tiny=False, download_file=False, use_cuda=False):

		device = torch.device('cpu')
		if use_cuda:
			device = torch.device('cuda')

		if use_tiny:
			cfg_file = 'files/yolov3-tiny.cfg'
			if download_file:
				download(use_tiny)
			# Set the location and name of the pre-trained weights file
			weight_file = 'files/yolov3-tiny.weights'
		else:
			cfg_file = 'files/yolov3.cfg'
			if download_file:
				download(use_tiny)
			# Set the location and name of the pre-trained weights file
			weight_file = 'files/yolov3.weights'

		# Set the location and name of the COCO object classes file
		namesfile = 'files/coco'

		# Load the network architecture
		self.m = Darknet(cfg_file, use_cuda).to(device)

		# Load the pre-trained weights
		self.m.load_weights(weight_file)

		# Load the COCO object classes
		self.class_names = load_class_names(namesfile)

		# Set the NMS threshold
		self.nms_thresh = 0.7
		# Set the IOU threshold
		self.iou_thresh = 0.3

	def draw_boxes(self, image, draw_im, labels = False):
		original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# We resize the image to the input width and height of the first layer of the network.    
		resized_image = cv2.resize(original_image, (self.m.width, self.m.height))
		# Detect objects in the image
		boxes = detect_objects(self.m, resized_image, self.iou_thresh, self.nms_thresh)
		#Plot the image with bounding boxes and corresponding object class labels
		box_img = plot_boxes(draw_im, boxes, self.class_names, labels)
		return box_img