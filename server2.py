import cv2
import flask
import numpy as np
import io  # Import io to use BytesIO

def server(stream_in, stream_out):
    app = flask.Flask(__name__)

    @app.route("/getFrameIn", methods=["GET"])
    def getFrameIn():
        ret, frame = stream_in.read()
        if not ret:
            return flask.make_response("error")
        ret, buffer = cv2.imencode(".png", frame)
        if not ret:
            return flask.make_response("error")
        buffer = buffer.tobytes()
        return flask.send_file(io.BytesIO(buffer), mimetype="image/png")

    @app.route("/getFrameOut", methods=["GET"])
    def getFrameOut():
        ret, frame = stream_out.read()
        if not ret:
            return flask.make_response("error")
        ret, buffer = cv2.imencode(".png", frame)
        if not ret:
            return flask.make_response("error")
        buffer = buffer.tobytes()
        return flask.send_file(io.BytesIO(buffer), mimetype="image/png")

    @app.route("/setMaskIn", methods=["POST"])
    def setMaskIn():
        request = flask.request.get_json()
        points = request["points"]
        threshold = request["threshold"]
        # get the current frame from the stream and get its shape, then draw the quad
        ret, frame = stream_in.read()
        if not ret:
            return flask.make_response("error")
        image = np.zeros(frame.shape, np.uint8)
        cv2.fillPoly(image, np.int32([points]), (255, 255, 255))
        cv2.imwrite("maskin.png", image)
        return flask.make_response("ok")

    @app.route("/setMaskOut", methods=["POST"])
    def setMaskOut():
        request = flask.request.get_json()
        points = request["points"]
        threshold = request["threshold"]
        ret, frame = stream_out.read()
        if not ret:
            return flask.make_response("error")
        image = np.zeros(frame.shape, np.uint8)
        cv2.fillPoly(image, np.int32([points]), (255, 255, 255))
        cv2.imwrite("maskout.png", image)
        return flask.make_response("ok")

    app.run()

if __name__ == "__main__":
    # for testing
    stream_in = cv2.VideoCapture(0)
    stream_out = cv2.VideoCapture(1)
    server(stream_in, stream_out)
