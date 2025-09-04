from flask import Flask, render_template, Response, request, jsonify
import cv2
from measurement import process_frame

app = Flask(__name__)


# Store camera index globally
camera_index = 0
cap = cv2.VideoCapture(camera_index)
pixels_per_cm = None
calibrated = False

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
        success, frame = cap.read()
        if not success:
            break
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
    camera_index = idx
    if cap:
        cap.release()
    cap = cv2.VideoCapture(camera_index)
    return jsonify({'success': True, 'camera_index': camera_index})

if __name__ == "__main__":
    app.run(debug=True)
