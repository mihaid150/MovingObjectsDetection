import cv2
import time
import os
from tracker import *
from ultralytics import YOLO
import math


def clean_folder(path):
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))


def window_conf():
    width = 640
    height = 480
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    cap.set(10, 150)
    return cap


def generate_video_frames(folder):
    clean_folder(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)

    cap = window_conf()

    frames = []
    start_time = time.time()

    # Capture time of 4 seconds
    while time.time() - start_time < 4:
        success, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(img_gray)

    cap.release()

    for i in range(len(frames)):
        cv2.imwrite(os.path.join(folder, f"frame_{i}.png"), frames[i])

    cv2.destroyAllWindows()
    return frames


def track_objects_simple():
    cap = window_conf()

    # Object detection from stable camera
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    tracker = EuclideanDistTracker()

    while True:
        success, img = cap.read()

        # Object detection
        mask = object_detector.apply(img)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cont in contours:
            # Compute area and remove small elements
            area = cv2.contourArea(cont)
            if area > 200:
                x, y, w, h = cv2.boundingRect(cont)
                detections.append([x, y, w, h])

        # Object tracking
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(img, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow("Live Cam", img)
        cv2.imshow("Mask", mask)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def track_objects_yolo():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1600)
    cap.set(4, 900)

    model = YOLO("yolo-Weights/yolov8n.pt")

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # Display top-right corner coordinates
                top_right_text = f"({x2}, {y1})"
                cv2.putText(img, top_right_text, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
