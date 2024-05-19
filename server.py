import cv2
import threading
import flask


def get_current_frame(stream):
    """Get the frame from the stream."""
    ret, frame = stream.read()
    return frame if ret else None


def make_frame_response(frame):
    """Create the response for the frame."""
    ret, buffer = cv2.imencode(".png", frame)
    return flask.make_response(buffer.tobytes())
 

def draw_quad(points, image_size, color=(255, 255, 255)):
    """Draw a quadrilateral on the image."""
    image = np.zeros(image_size, np.uint8)
    cv2.fillPoly(image, np.int32([points]), color)
    return image


def save_to_file(image, filename):
    """Save the image to a file."""
    cv2.imwrite(filename, image)


def make_counter(in_stream, out_stream):
    pass
 

def make_server(in_stream, out_stream):
    """Create the webserver."""
    app = flask.Flask(__name__)
    count = 0

    @app.route("/frame/in", methods=["GET"])
    def video_in():
        """Get the frame from the stream."""
        return make_frame_response(get_current_frame(in_stream))

    @app.route("/frame/out", methods=["GET"])
    def video_out():
        """Get the frame from the stream."""
        return make_frame_response(get_current_frame(out_stream))

    @app.route("/mask/in", methods=["POST"])
    def mask_in():
        request = flask.request.get_json()
        points = request["points"]
        threshold = request["threshold"]
        image_size = get_current_frame(in_stream).shape
        image = draw_quad(points, image_size)
        save_to_file(image, "maskin.png")
        return flask.make_response("ok")

    @app.route("/mask/out", methods=["POST"])
    def mask_out():
        request = flask.request.get_json()
        points = request["points"]
        image_size = get_current_frame(out_stream).shape
        image = draw_quad(points, image_size)
        save_to_file(image, "maskout.png")
        return flask.make_response("ok")

    def start_server():
        app.run(debug=True)

    return start_server
