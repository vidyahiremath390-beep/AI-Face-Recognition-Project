from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
import database
from train_model import train

app = Flask(__name__)
database.init_db()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists("trainer/trainer.yml"):
    recognizer.read("trainer/trainer.yml")

current_user_id = None
capture_count = 0


def generate_frames():
    global current_user_id, capture_count

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(100,100))

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))

                if os.path.exists("trainer/trainer.yml") and os.path.getsize("trainer/trainer.yml") > 0:
                    try:
                        id, confidence = recognizer.predict(face_img)
                    except:
                        id, confidence = None, 100

                    if confidence < 80:
                        name = database.get_user_name(id)
                    else:
                        name = "New User"
                else:
                    name = "New User"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=["GET", "POST"])
def register():
    global recognizer

    if request.method == "POST":
        name = request.form["name"]
        user_id = database.insert_user(name)

        user_path = f"dataset/{user_id}"
        os.makedirs(user_path, exist_ok=True)

        count = 0
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            return "Unable to access camera", 500

        try:
            while count < 20:
                ret, frame = capture.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    if count >= 20:
                        break

                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (200, 200))
                    count += 1
                    file_path = f"{user_path}/{count}.jpg"
                    cv2.imwrite(file_path, face_img)

                    cv2.imshow("Capturing Face", face_img)
                    cv2.waitKey(100)

            cv2.destroyAllWindows()
        finally:
            capture.release()
            cv2.destroyAllWindows()

        # Train model
        train()

        # Reload model
        recognizer.read("trainer/trainer.yml")

        return redirect(url_for('index'))

    return render_template("register.html")


if __name__ == "__main__":
    app.run(debug=True)