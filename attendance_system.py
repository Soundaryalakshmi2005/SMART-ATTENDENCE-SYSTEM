import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# Initialize Camera
video_capture = cv2.VideoCapture(0)  # Change index if needed
if not video_capture.isOpened():
    print("⚠️ Camera not detected. Please check your camera connection and index.")
    exit()

# Define CSV file
CSV_FILE = "attendance.csv"

# Ensure the CSV file exists and has correct headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "roll_no", "year", "department", "section", "date", "time", "status"])  # Define required columns

# Load known student images and encodings
known_face_encodings = []
known_face_names = []
students_data = {}
image_dir = "c:/HTML/AttendanceSystem/images/"

for file in os.listdir(image_dir):
    if file.endswith(".jpg") or file.endswith(".png"):
        student_name = os.path.splitext(file)[0]
        image_path = os.path.join(image_dir, file)
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encoding = np.mean(rgb_image, axis=(0, 1))  # Improved encoding logic
        known_face_encodings.append(face_encoding)
        known_face_names.append(student_name)
        students_data[student_name] = {
            "rollno": "ID" + student_name[-3:], 
            "year": "III", 
            "department": "CSE", 
            "section": "A"
        }

# Function to check if attendance is already marked
def is_attendance_marked(name, date):
    try:
        with open(CSV_FILE, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["name"] == name and row["date"] == date:
                    return True
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return False

# Function to add attendance record
def add_attendance(name, rollno, year, department, section, date, time, status):
    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, rollno, year, department, section, date, time, status])

# Main loop for face detection and attendance marking
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("⚠️ Camera not detected. Check camera index.")
        break
    
    # Convert frame to RGB (Mediapipe expects RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    # If faces detected, process them
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            
            # Extract face region
            face_region = rgb_frame[y:y+height, x:x+width]
            if face_region.size == 0:
                continue
            
            face_encoding = np.mean(face_region, axis=(0, 1))  # Improved encoding logic
            
            # Compare with known encodings
            matches = [np.linalg.norm(face_encoding - enc) for enc in known_face_encodings]
            best_match_index = np.argmin(matches) if matches else None
            
            if best_match_index is not None and matches[best_match_index] < 5000:  # Adjusted threshold
                name = known_face_names[best_match_index]
                student = students_data.get(name, {})
                rollno = student.get("rollno", "-")
                year = student.get("year", "-")
                department = student.get("department", "-")
                section = student.get("section", "-")
                status = "Present"

                # Check if attendance is already marked for today
                current_date = datetime.now().strftime("%Y-%m-%d")
                if not is_attendance_marked(name, current_date):  
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    add_attendance(name, rollno, year, department, section, current_date, current_time, status)

                    # Draw bounding box around detected face
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} - Present", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the frame with detected faces
    cv2.imshow("Smart Attendance System - Mediapipe", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
