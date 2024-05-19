import numpy as np
from ultralytics import YOLO
import cv2
import threading
import math
import flask
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

def init_mask(stream):
    ret, frame = stream.read()
    if not ret:
        return None
    mask_shape = frame.shape
    mask = np.full_like(mask_shape, 255, np.uint8)
    return mask

def server(stream):
    app = flask.Flask(__name__)
    config = {
        'limits_in': [70, 170, 160, 170],
        'limits_out': [120, 60, 200, 60],
        'mask': init_mask(stream)
    }


    def protector_server():
        model = YOLO("yolov8l.pt")
        tracker = Sort(max_age=60)

        max_count = 3
        soft_limit = 1
        escalator_controller(stream, config, model, tracker, max_count, soft_limit)
        stream.release()


    @app.route("/frame", methods=["GET"])
    def getFrameIn():
        ret, frame = stream.read()
        if not ret:
            return flask.make_response("error")
        ret, buffer = cv2.imencode(".png", frame)
        if not ret:
            return flask.make_response("error")
        buffer = buffer.tobytes()
        return flask.send_file(io.BytesIO(buffer), mimetype="image/png")

    @app.route("/config", methods=["POST"])
    def setMaskIn():
        request = flask.request.get_json()
        points = request["points"]
        config['limits_in'] = [*request["line1Points"][0].values(), *request["line1Points"][1].values()]
        config['limits_out'] = [*request["line2Points"][0].values(), *request["line2Points"][1].values()]
        # get the current frame from the stream and get its shape, then draw the quad
        ret, frame = stream.read()
        if not ret:
            return flask.make_response("error")
        image = np.zeros(frame.shape, np.uint8)
        cv2.fillPoly(image, np.int32([points]), (255, 255, 255))
        config['mask'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return flask.make_response("ok")


    def start_server():
        threading.Thread(target=protector_server).start()
        app.run()

    return start_server

cap = cv2.VideoCapture("videos/sample.mp4")
start = server(cap)

start()