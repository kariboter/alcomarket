import argparse
import time
import cv2
import imutils
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

import triangulation as tri

from gpiozero import PWMLED
from gpiozero.pins.pigpio import PiGPIOFactory

factory = PiGPIOFactory(host='192.168.88.226')
PWM_l = PWMLED(17, pin_factory=factory)
DIR_l = PWMLED(4, pin_factory=factory)
PWM_r = PWMLED(3, pin_factory=factory)
DIR_r = PWMLED(2, pin_factory=factory)

DIR_r.off()
from PID import PID
pid = PID(0.25, 0, 0)
pid.sample_time = 0
pid.SetPoint = 200

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--proto_txt", required=True,
                help="path to Caffe 'deploy' proto_txt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video_source", type=int, default=0,
                help="video source (default = 0, external usually = 1)")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "dining_table",
           "dog", "horse", "motorbike", "person", "potted_plant", "sheep",
           "sofa", "train", "tv_monitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["proto_txt"], args["model"])

# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src='http://192.168.88.226:9000/stream.mjpg').start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()

B = 11  # Distance between the cameras [cm]
alpha = 60  # Camera field of view in the horisontal plane [degrees]
speed_K = (0.25, 0.1, 0.01)
rotation_K = (1, 0, 0)
center_right = 0
center_left = 0
width, height = (640, 240)
half_width = 320
percentage_l = 0.0
percentage_r = 0.0
idx_l = 0
idx_r = 0

def feedback_calc(center_r, center_l):
    feedback = center_r - (400 - center_l) + 200
    return feedback


def rotation(feedback, Ks):
        Kp, Ki, Kd = Ks
        pid.set_Ks(Kp, Ki, Kd)
        pid.update(feedback)
        return pid.output


def motors(rotation, percentage_l, percentage_r):
        if (pid.delta_time >= pid.sample_time):
            if percentage_l <= 0.8 and percentage_r <= 0.8:
                PWM_l.value = 0.5 - ((rotation * 0.2) / 50)
                PWM_r.value = 0.5 + ((rotation * 0.2) / 50)
            elif rotation >= 190:
                PWM_l.value = 0.6 - ((rotation * 0.2) / 50)
                PWM_r.value = 0.6 + ((rotation * 0.2) / 50)
            else:
                PWM_l.value = 0
                PWM_r.value = 0
def depth(center_right, center_left, frame_r, frame_l, B, alpha, percentage_l, percentage_r):
    if percentage_l <= 0.7 and percentage_r <= 0.7:
        depth = tri.find_depth(center_right, center_left, frame_r, frame_l, B, alpha)
        return depth

while True:

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    img_r = frame[0:height, 0:half_width]
    img_l = frame[0:height, half_width:width]
    frame_r = imutils.resize(img_r, width=400)
    frame_l = imutils.resize(img_l, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame_r.shape[:2]
    blob_r = cv2.dnn.blobFromImage(cv2.resize(frame_r, (300, 300)),
                                   0.007843, (300, 300), 127.5)
    blob_l = cv2.dnn.blobFromImage(cv2.resize(frame_l, (300, 300)),
                                   0.007843, (300, 300), 127.5)
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob_r)
    detections_right = net.forward()
    net.setInput(blob_l)
    detections_left = net.forward()

    # loop over the detections
    for i in np.arange(0, detections_right.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections_right[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx_r = int(detections_right[0, 0, i, 1])
            if idx_r == 15:
                box = detections_right[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx_r],
                                             confidence * 100)

                cv2.rectangle(frame_r, (startX, startY), (endX, endY),
                              COLORS[idx_r], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame_r, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx_r], 2)
                center_right = startX + ((endX - startX) / 2)
                percentage_r = (endX - startX) / 400
    for i in np.arange(0, detections_left.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections_left[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx_l = int(detections_left[0, 0, i, 1])
            if idx_l == 15:
                box = detections_left[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx_l],
                                             confidence * 100)

                cv2.rectangle(frame_l, (startX, startY), (endX, endY),
                              COLORS[idx_l], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame_l, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx_l], 2)
                center_left = startX + ((endX - startX) / 2)
                percentage_l = (endX - startX) / 400


    # show the output frame
    cv2.imshow("Frame_r", frame_r)
    cv2.imshow("Frame_l", frame_l)
    motors(rotation(feedback_calc(center_right, center_left),speed_K), percentage_l, percentage_r)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()