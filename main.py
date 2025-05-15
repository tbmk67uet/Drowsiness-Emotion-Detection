import tkinter as tk
from tkinter import ttk
from threading import Thread
from datetime import datetime
import os
import cv2
from PIL import Image, ImageTk
import csv
import numpy as np
import time
from view_results import view_results_from_csv

from keras.models import load_model
from keras.utils import img_to_array
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
from pygame import mixer

# === GLOBALS ===
app_running = False
vs = None
log_file = None
log_writer = None
log_path = None
frame_label = None
drowsy_display_frames = 0
yawn_display_frames = 0
alarm_status = False
alarm_status2 = False
COUNTER = 0
YAWN_COUNTER = 0

# === INIT MODELS ===
mixer.init()
sound1 = mixer.Sound('wake_up.mp3')
sound2 = mixer.Sound('alert.mp3')

emotion_model = load_model('model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# === METRICS ===
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 10
YAWN_THRESH = 30
YAWN_CONSEC_FRAMES = 15

# === FUNCTIONS ===
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return ((leftEAR + rightEAR) / 2.0, leftEye, rightEye)

def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1])

def start_session():
    global app_running, vs, log_file, log_writer, log_path, COUNTER, YAWN_COUNTER, alarm_status, alarm_status2
    if app_running:
        return
    app_running = True
    tab_control.select(tab1)

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"emotion_log_{timestamp}.csv")
    log_file = open(log_path, mode="w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Timestamp", "Emotion", "EAR", "YawnDistance", "DrowsinessAlert", "YawnAlert"])

    vs = cv2.VideoCapture(0)
    COUNTER = 0
    YAWN_COUNTER = 0
    alarm_status = False
    alarm_status2 = False

    update_video()

def end_session():
    global app_running, vs, log_file, log_path
    app_running = False
    if vs:
        vs.release()
    if log_file:
        log_file.close()
    tab_control.select(tab2)
    view_results_from_csv(tab2, log_path)

def update_video():
    global app_running, vs, frame_label, log_writer, drowsy_display_frames, yawn_display_frames, alarm_status, alarm_status2, COUNTER, YAWN_COUNTER

    if not app_running:
        return

    ret, frame = vs.read()
    if not ret:
        return

    frame = cv2.resize(frame, (500, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(face_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = emotion_model.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]

        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = landmark_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, _, _ = final_ear(shape)
        lip_dist = lip_distance(shape)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES and not alarm_status:
                alarm_status = True
                Thread(target=sound1.play, daemon=True).start()
                drowsy_display_frames = 30
        else:
            COUNTER = 0
            alarm_status = False

        if drowsy_display_frames > 0:
            cv2.putText(frame, "DROWSINESS ALERT!", (frame.shape[1] - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            drowsy_display_frames -= 1

        if lip_dist > YAWN_THRESH:
            YAWN_COUNTER += 1
            if YAWN_COUNTER >= YAWN_CONSEC_FRAMES and not alarm_status2:
                alarm_status2 = True
                Thread(target=sound2.play, daemon=True).start()
                yawn_display_frames = 30
        else:
            YAWN_COUNTER = 0
            alarm_status2 = False

        if yawn_display_frames > 0:
            cv2.putText(frame, "YAWN ALERT!", (frame.shape[1] - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yawn_display_frames -= 1

        drowsy_flag = "Yes" if alarm_status else "No"
        yawn_flag = "Yes" if alarm_status2 else "No"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_writer.writerow([ts, label, f"{ear:.2f}", f"{lip_dist:.2f}", drowsy_flag, yawn_flag])

        cv2.putText(frame, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"YAWN: {lip_dist:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(img))
    frame_label.imgtk = img
    frame_label.configure(image=img)

    tab1.after(10, update_video)

# === GUI ===
root = tk.Tk()
root.title("Focus Tracker App")
root.geometry("900x650")

tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)

tab_control.add(tab1, text='📷 Start Session')
tab_control.add(tab2, text='📊 View Current Result')
tab_control.add(tab3, text='📁 Past Sessions')
tab_control.pack(expand=1, fill='both')

# Tab 1: Start Session
btn_start = tk.Button(tab1, text="▶️ Start Session", command=start_session, font=("Arial", 14))
btn_start.pack(pady=10)

btn_stop = tk.Button(tab1, text="⏹️ End Session", command=end_session, font=("Arial", 14))
btn_stop.pack(pady=10)

frame_label = tk.Label(tab1)
frame_label.pack(padx=10, pady=10)

# Tab 3: List of past logs
def list_logs():
    logs = sorted(os.listdir("logs"))
    listbox.delete(0, tk.END)
    for log in logs:
        listbox.insert(tk.END, log)

def on_log_select(event):
    selection = event.widget.curselection()
    if selection:
        index = selection[0]
        filename = event.widget.get(index)
        filepath = os.path.join("logs", filename)
        tab_control.select(tab2)
        view_results_from_csv(tab2, filepath)

listbox = tk.Listbox(tab3, width=80)
listbox.pack(pady=20)
listbox.bind('<<ListboxSelect>>', on_log_select)

btn_refresh = tk.Button(tab3, text="🔄 Refresh List", command=list_logs)
btn_refresh.pack()

list_logs()

root.mainloop()
