import numpy as np
from ultralytics import YOLO
import cv2
import math
from sort import Sort
from helper import create_video_writer


def is_in_limits(limits, cx, cy, k=15):
    x1, y1, x2, y2 = limits
    return x1 < cx < x2 and y1 - k < cy < y2 + k


def count_people(img, mask, model, tracker):
    imgRegion = cv2.bitwise_and(img, mask)
    results = model.track(imgRegion, persist=True)
    detections = np.empty((0, 5))
    for r in results:
        for box in r.boxes:
            # Confidence
            conf = float(box.conf[0])  # math.ceil((box.conf[0] * 100)) / 100
            if int(box.cls[0]) != 0 or conf < 0.1:
                continue
            # Bounding Box
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))

    return tracker.update(detections)


def gate_action(gate_closed, count, max_count, soft_limit):
    if count > max_count:
        # close gate
        print("CLOSING GATE")
        return True
    if gate_closed and (count < soft_limit):
        # open gate
        print("OPENING GATE")
        return False
    return gate_closed

def draw_line(img, xyxy, color=(0, 0, 255), thickness=5):
    x1, y1, x2, y2 = [int(i) for i in xyxy]
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def make_ui_frame(img, frame_size, gate_closed):
    frame = np.zeros((frame_size[1], frame_size[0], 3), np.uint8)
    img_shape = img.shape
    frame[0:img.shape[0], 0:img.shape[1]] = img
    # if gate closed, draw white rectangle right of img till end. else black
    color = (0, 0, 0) if gate_closed else (255, 255, 255)
    frame[0:img.shape[0], img.shape[1]:] = color
    return frame


def escalator_controller(
    stream, limits_in, limits_out, mask_in, mask_out, model, tracker, soft_limit
):
    people = []
    gate_closed = False
    while stream.isOpened():
        success, img = cap.read()
        if not success:
            break
        results = count_people(img, mask, model, tracker)
        draw_line(img, limits_in)
        draw_line(img, limits_out)
        for result in results:
            x1, y1, x2, y2, _id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            if is_in_limits(limits_in, cx, cy, 15) and people.count(_id) == 0:
                people.append(_id)
            if is_in_limits(limits_out, cx, cy, 15) and people.count(_id) == 1:
                people.remove(_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        count = len(people)
        gate_closed = gate_action(gate_closed, count, max_count, soft_limit)
        if (not gate_closed) and (count >= soft_limit):
            # sound alarm
            print("ALARM")
        cv2.putText(
            img,
            f"Count: {count}",
            (480, 40),
            1,
            2,
            (139, 195, 75),
            3,
        )
        cv2.imshow("Video", make_ui_frame(img, (1280, img.shape[0]), gate_closed))
        if cv2.waitKey(1) == ord("q"):
            break


cap = cv2.VideoCapture("videos/sample.mp4")  # For Video
model = YOLO("yolov8l.pt")
mask = cv2.imread("Images/mask.png")
tracker = Sort(max_age=60)
limits_in = [70, 170, 160, 170]
limits_out = [120, 60, 200, 60]
max_count = 3
soft_limit = 1
escalator_controller(cap, limits_in, limits_out, mask, mask, model, tracker, soft_limit)
cap.release()
cv2.destroyAllWindows()
