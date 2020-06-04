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
#MIN_DIST = 50 # min distance in pixels
#Link to video file and where to output

def calibrated_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + 550 / ((p1[1] + p2[1]) / 2) * (p1[1] - p2[1]) ** 2) ** 0.5


def isclose(p1, p2):
    c_d = calibrated_dist(p1, p2)
    calib = (p1[1] + p2[1]) / 2
    if 0 < c_d < 0.15 * calib:
        return 1
    elif 0 < c_d < 0.2 * calib:
        return 2
    else:
        return 0

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
                    centroids.append([centerX, centerY])
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

    status = [0]*len(results)
    close_pair = list()
    s_close_pair = list()

    if len(results) >= 2:
        #we extract all centroid from the results and compute the
      
        centroids = np.array([r[2] for r in results])

        for i in range(len(centroids)):
            for j in range(len(centroids)):

                dist = isclose(centroids[i], centroids[j])

                if dist  == 1:
                    close_pair.append([centroids[i], centroids[j]])
                    status[i] = 1
                    status[j] = 1
                elif dist == 2:
                    s_close_pair.append([centroids[i], centroids[j]])          
                    if status[i] != 1:
                        status[i] = 2
                    if status[j] != 1:
                        status[j] = 2

    low_risk_p = status.count(2)
    high_risk_count = status.count(1)
    safe_p = status.count(0)          

    iter = 0 
    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        
        if status[iter] == 1:
            color = (0, 0, 255)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        elif status[iter] == 0:
            color = (0, 255, 0)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        else:
            color = (0, 165, 255)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
        iter += 1

    # Connecting Lines
    for i in close_pair:
        cv2.line(frame, tuple(i[0]), tuple(i[1]), (0,0, 255), 2)
    for i in s_close_pair:
        cv2.line(frame, tuple(i[0]), tuple(i[1]), (0,255, 255), 2)

    text = "Total Violations: {}".format(high_risk_count+low_risk_p)
    cv2.putText(frame, text, (10, frame.shape[0] -25) , cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0, 255), 4)

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
