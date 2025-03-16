from flask import Flask, render_template, jsonify, send_file
import csv
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

CSV_FILE = "attendance.csv"

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# Ensure CSV file exists
def initialize_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Roll No", "Year", "Department", "Section", "Date", "Time", "Status"])

initialize_csv()

# Load attendance records from CSV
def get_attendance():
    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        data = list(reader)[1:]  # Skip header row
    return data

@app.route("/")
def index():
    return render_template("index.html", data=get_attendance())

@app.route("/data")
def data():
    return jsonify(get_attendance())

@app.route("/download")
def download():
    return send_file(CSV_FILE, as_attachment=True)

# Load student images & encodings
def load_known_faces(image_dir="images/"):
    known_encodings, known_names = [], {}
    for file in os.listdir(image_dir):
        if file.endswith((".jpg", ".png")):
            name = os.path.splitext(file)[0]
            image_path = os.path.join(image_dir, file)
            image = cv2.imread(image_path)
            face_encoding = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=(0, 1))
            known_encodings.append(face_encoding)
            known_names[name] = {"rollno": "ID" + name[-3:], "year": "III", "department": "CSE", "section": "A"}
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

# Mark attendance in CSV (Ensure one entry per student per day)
def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    # Read existing attendance to check if the student is already marked present
    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        existing_data = list(reader)

    for row in existing_data:
        if row[0] == name and row[5] == today:  # Check if student is already marked present today
            return  # Exit function without adding duplicate attendance

    student = known_names.get(name, {})
    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, student.get("rollno", "-"), student.get("year", "-"), student.get("department", "-"),
                         student.get("section", "-"), today, time_now, "Present"])

# Face Recognition & Attendance Marking
def recognize_faces():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ Camera not detected.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                face_region = rgb_frame[y:y+height, x:x+width]
                if face_region.size == 0:
                    continue
                face_encoding = np.mean(face_region, axis=(0, 1))
                matches = [np.linalg.norm(face_encoding - enc) for enc in known_encodings]
                best_match_index = np.argmin(matches) if matches else None
                if best_match_index is not None and matches[best_match_index] < 5000:
                    name = list(known_names.keys())[best_match_index]
                    mark_attendance(name)
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} - Present", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Smart Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(debug=True)
