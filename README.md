# Social Distancing Analyzer

Analyzes the distance between close people in a given video feed. Human detection is done using YOLOv3 model

## File structure

You will need the YOLOv3 models and COCO dataset to run the file. Your file structure should look like this -

```tree
|---yolo-coco
|   |--coco.names
|   |--yolov3.weights
|   |--yolov3.cfg
|---output
|---video
|---peds_detection.py
```
## Other Dependencies

OpenCv4.0, Numpy, and imutils are needed.

## Usage

```
usage: peds_detection.py [-h] [-i INPUTDIRECTORY] [-o OUTPUTDIRECTORY]

Path to input and output directory

optional arguments:
  -h, --help            show this help message and exit
  -i INPUTDIRECTORY, --inputDirectory INPUTDIRECTORY
                        Path to the input directory
  -o OUTPUTDIRECTORY, --outputDirectory OUTPUTDIRECTORY
                        Path to the output
```
## Credit
[Adrain Rosebrock](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/) for object detection post and [Ankush Chaudhari](https://www.linkedin.com/in/ankush-chaudhari/) for the function which measures the distance between two people.