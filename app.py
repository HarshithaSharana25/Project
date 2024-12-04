from flask import Flask, render_template, Response, jsonify
import cv2
from mediapipe_test import PostureDetector
import os
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
detector = None
camera = None
should_stop = False

# Ensure static/graphs directory exists
os.makedirs('static/graphs', exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')


def generate_frames():
    global detector, camera, should_stop
    
    if camera is None:
        camera = cv2.VideoCapture(0)
    if detector is None:
        detector = PostureDetector(angle_threshold=10)
    
    try:
        while not should_stop:
            success, frame = camera.read()
            if not success:
                break
            
            analyzed_frame = detector.analyze_posture(frame)
            ret, buffer = cv2.imencode('.jpg', analyzed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except:
        if camera is not None:
            camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_posture')
def start_posture():
    global should_stop
    should_stop = False
    return render_template('posture.html')

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detector, camera, should_stop
    should_stop = True
    
    if detector is not None:
        detector.save_plots('static/graphs')
        detector = None
    
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(debug=True)