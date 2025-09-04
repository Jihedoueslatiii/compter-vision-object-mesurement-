from flask import Flask, render_template, Response, request, jsonify
import cv2
from measurement import process_frame
import threading
import os

app = Flask(__name__)

# Store camera index globally
camera_index = 0
cap = None
cap_lock = threading.Lock()
pixels_per_cm = None
calibrated = False

def open_camera(idx):
    cam = cv2.VideoCapture(idx)
    if not cam.isOpened():
        return None
    return cam

# Detect available cameras
def list_cameras(max_tested=5):
    available = []
    for i in range(max_tested):
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            available.append(i)
            test_cap.release()
    return available

def generate_frames():
    global pixels_per_cm, calibrated, cap
    while True:
        with cap_lock:
            if cap is None:
                cap = open_camera(camera_index)
            if cap is None:
                # Show error frame
                import numpy as np
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, 'Camera not available', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue
            success, frame = cap.read()
        if not success or frame is None:
            # Show error frame
            import numpy as np
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, 'Camera read failed', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue
        frame, calibrated, pixels_per_cm, w, h = process_frame(frame, calibrated, pixels_per_cm)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    cameras = list_cameras()
    return render_template('index.html', cameras=cameras, selected_camera=camera_index)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API to change camera
@app.route('/set_camera', methods=['POST'])
def set_camera():
    global cap, camera_index
    idx = int(request.form.get('camera_index', 0))
    with cap_lock:
        camera_index = idx
        if cap:
            cap.release()
        cap = open_camera(camera_index)
    success = cap is not None and cap.isOpened()
    return jsonify({'success': success, 'camera_index': camera_index})

# Vercel handler
def handler(request):
    return app(request.environ, lambda *args: None)

if __name__ == "__main__":
    app.run(debug=True)