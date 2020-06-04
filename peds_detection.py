import numpy as np
import imutils
import time
import cv2
import sys
import os
import argparse
from pathlib import Path
from scipy.spatial import distance as dist

confid = 0.5
thresh = 0.5
#Link to video file and where to output
def create_arg_parser():
    parser = argparse.ArgumentParser(description="Path to input and output directory")

    parser.add_argument('-i', '--inputDirectory', help='Path to the input directory', type=Path)
    parser.add_argument('-o', '--outputDirectory', help = 'Path to the output', type=Path)
    return parser

arg_parser = create_arg_parser()
parsed_args = arg_parser.parse_args(sys.argv[1:])
 


labelsPath = "yolo-coco//coco.names"
#LABELS contains all the types of items which can be detected in the form of a list
LABELS =  open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = 'yolo-coco/yolov3.weights'
configPath = 'yolo-coco/yolov3.cfg'




#Contains all the layer names of network
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


vs = cv2.VideoCapture(str(parsed_args.inputDirectory))
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None
(W, H) = (None, None)
########################################################################
#For time
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1
############################################################################

while True:
    (grabbed, frame) =vs.read()

    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []
    centroids = []
    results = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if LABELS[classID] == "person":
                if confidence > confid:

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    centroids.append((centerX, centerY))
                    classIDs.append(classID)


    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
#Drawing boxes and GUI part starts from here for measuring 
##############################################################################
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            r = (confidences[i], (x,y,x+w, y+h), centroids[i])
            results.append(r)

    if writer is None:
        writer = cv2.VideoWriter(str(parsed_args.outputDirectory), fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

    writer.write(frame)

print("[INFO] cleaning up...")
writer.release()
vs.release()
