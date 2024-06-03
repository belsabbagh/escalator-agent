import numpy as np
from ultralytics import YOLO
import cv2
import sys
import threading
import math
import flask
import pygame
import time
from sort import Sort
from PyQt6 import QtCore, QtGui, QtWidgets

pygame.mixer.init()
alert = pygame.mixer.Sound('assets/beep-warning-6387.mp3')

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
    if count > max_count and not gate_closed:
        # close gate
        alert.play()
        time.sleep(1)
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
    stream, config, model, tracker, max_count, soft_limit
):
    people = []
    gate_closed = False
    while stream.isOpened():
        success, img = stream.read()
        if not success:
            break
        results = count_people(img, config['mask'], model, tracker)
        draw_line(img, config['limits_in'])
        draw_line(img, config['limits_out'])
        for result in results:
            x1, y1, x2, y2, _id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            if is_in_limits(config['limits_in'], cx, cy, 15) and people.count(_id) == 0:
                people.append(_id)
            if is_in_limits(config['limits_out'], cx, cy, 15) and people.count(_id) == 1:
                people.remove(_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        count = len(people)
        gate_closed = gate_action(gate_closed, count, max_count, soft_limit)
        if (not gate_closed) and (count >= soft_limit):
            cv2.putText(
                img,
                f"Remaining: {max_count - count}",
                (420, 70),
                1,
                2,
                (50, 50, 230),
                3,
            )
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



class DisplayImageWidget(QtWidgets.QWidget):
    def __init__(self, image):
        super(DisplayImageWidget, self).__init__()
        self.image = image
        self.points = []

        self.convert = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], QtGui.QImage.Format.Format_BGR888)
        self.frame = QtWidgets.QLabel()
        self.pixmap = QtGui.QPixmap.fromImage(self.convert)
        self.frame.setPixmap(self.pixmap)
        self.resize(self.pixmap.width(), self.pixmap.height())
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self.frame)

        self.frame.mousePressEvent = self.get_pos

    def get_pos(self, event):
        if len(self.points) < 8:
            x = event.pos().x() - self.pos().x()
            y = event.pos().y() - self.pos().y()
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")

            # Draw the points on the image
            painter = QtGui.QPainter(self.convert)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 5))
            painter.drawPoint(x, y)
            painter.end()

            self.frame.setPixmap(QtGui.QPixmap.fromImage(self.convert))

        if len(self.points) == 8:
            config = {}
            print("All points selected:", self.points)
            mask_points = np.array(self.points[:4])
            mask = np.zeros(self.image.shape, np.uint8)
            cv2.fillPoly(mask, np.int32([mask_points]), (255, 255, 255))
            config['mask'] = mask
            config['limits_in'] = np.array([*self.points[4], *self.points[5]])
            config['limits_out'] = np.array([*self.points[6], *self.points[7]])
            model = YOLO("yolov8l.pt")
            tracker = Sort(max_age=60)

            max_count = 3
            soft_limit = 1
            self.close()
            escalator_controller(stream, config, model, tracker, max_count, soft_limit)

            stream.release()
            

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Static Image with PyQt")
        height, width, _ = image.shape
        self.setGeometry(100, 100, width, height)
        self.image_widget = DisplayImageWidget(image)
        self.setCentralWidget(self.image_widget)



if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    # Capture a single frame from the video stream
    stream = cv2.VideoCapture("http://192.168.64.254:4747/video")
    ret, frame = stream.read()

    if not ret:
        print("Failed to capture image")
        sys.exit(1)

    main_window = MainWindow(frame)
    main_window.show()
    sys.exit(app.exec())