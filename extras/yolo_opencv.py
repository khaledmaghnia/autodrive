import numpy as np
import cv2
import time
import os
import signal

class YOLOMODEL():
	def __init__(self):
		self.confThresh = 0.5
		self.nmsThresh = 0.3


		classPath = 'files/coco'
		classPathTiny = 'files/coco.names'
		self.LABELS = open(classPath, 'r').read().strip().split('\n')
		self.LABELS_TINY = open(classPathTiny, 'r').read().strip().split('\n')


		np.random.seed(42)
		self.COLORS = np.random.randint(0,255, size=(len(self.LABELS), 3), dtype=np.uint8)


		weightsPath = 'files/yolov3.weights'
		cfgPath = 'files/yolov3.cfg'

		self.net = cv2.dnn.readNetFromDarknet(cfgPath, weightsPath)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

	def yolo_forward(self, image):
		self.boxes=[]
		self.confidences=[]
		self.classIDs = []

		(self.H,self.W) = image.shape[:2]

		ln = self.net.getLayerNames()
		ln = [ln[i[0]-1] for i in self.net.getUnconnectedOutLayers()]

		blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
		self.net.setInput(blob)
		self.layerOutputs=self.net.forward(ln)

		self.get_idxs()

	def get_idxs(self):
		for output in self.layerOutputs:
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				if self.LABELS[classID] in self.LABELS_TINY:
					if confidence > self.confThresh:
						box = detection[0:4]*np.array([self.W,self.H,self.W,self.H])
						(centerX, centerY, width, height) = box.astype('int')

						x = int(centerX - (width/2))
						y = int(centerY - (height/2))

						self.boxes.append([x, y, int(width), int(height)])
						self.confidences.append(float(confidence))
						self.classIDs.append(classID)

		self.idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.confThresh, self.nmsThresh)

	def annotate(self, image):
		if len(self.idxs)>0:
			for i in self.idxs.flatten():
				(x, y) = (self.boxes[i][0], self.boxes[i][1])
				(w, h) = (self.boxes[i][2], self.boxes[i][3])

				color = [int(c) for c in self.COLORS[self.classIDs[i]]]
				cv2.rectangle(image, (x,y), (x+w, y+h), color, 2)
				text = f'{self.LABELS[self.classIDs[i]]}: {round(self.confidences[i], 2)}'
				(label_width, label_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
				cv2.putText(image, text, (x+int(w/2)-int(label_width/2), y-label_height+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		return image