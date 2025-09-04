from flask import Flask, render_template, Response
import cv2
from measurement import process_frame

app = Flask(__name__)

cap = cv2.VideoCapture(0)
pixels_per_cm = None
calibrated = False

def generate_frames():
    global pixels_per_cm, calibrated
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
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
