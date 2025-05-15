import customtkinter as ctk
from tkinter import ttk
from threading import Thread
from datetime import datetime
import os
import cv2
from PIL import Image, ImageTk
import csv
import numpy as np
import time
from view_results import view_results_from_csv, export_figure_to_pdf
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
countdown_seconds = None
countdown_label = None
drowsy_display_frames = 0
yawn_display_frames = 0
alarm_status = False
alarm_status2 = False
alarm_status3 = False
COUNTER = 0
YAWN_COUNTER = 0
FACE_MISSING_COUNTER = 0
paused = False
start_time = None
paused_time = 0
pause_start = None

# === INIT MODELS ===
mixer.init()
sound1 = mixer.Sound('wake_up.mp3')
sound2 = mixer.Sound('alert.mp3')
sound3 = mixer.Sound('Where Are You, Sir_.mp3')

emotion_model = load_model('model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# === METRICS ===
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 10
YAWN_THRESH = 30
YAWN_CONSEC_FRAMES = 15
FACE_MISSING_THRESHOLD = 50

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
    global app_running, vs, log_file, log_writer, log_path, countdown_seconds, COUNTER, YAWN_COUNTER, FACE_MISSING_COUNTER, alarm_status, alarm_status2, alarm_status3
    if app_running:
        return
    app_running = True
    global start_time
    start_time = time.time()

    tabview.set("\ud83d\udcf7 Start Session")

    start_btn.pack_forget()
    timer_entry.pack_forget()
    timer_label.pack_forget()
    time_label.pack_forget()

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"emotion_log_{timestamp}.csv")
    log_file = open(log_path, mode="w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Timestamp", "Emotion", "EAR", "YawnDistance", "DrowsinessAlert", "YawnAlert", "FaceMissing"])

    vs = cv2.VideoCapture(0)
    COUNTER = 0
    YAWN_COUNTER = 0
    FACE_MISSING_COUNTER = 0
    alarm_status = False
    alarm_status2 = False
    alarm_status3 = False

    timer_val = timer_entry.get()
    countdown_seconds = int(timer_val) * 60 if timer_val.strip().isdigit() else None
    countdown_label.place(relx=0.01, rely=0.95, anchor='sw')

    if countdown_seconds:
        countdown_label.place(relx=0.01, rely=0.95, anchor='sw')
        update_countdown()
    else:
        countdown_label.place_forget()
        time_label.pack(pady=10)
        update_timer()

    update_video()

def update_countdown():
    global countdown_seconds, countdown_label, app_running

    if not app_running:
        return
    if paused:
        tab1.after(1000, update_countdown)
        return

    if countdown_seconds is not None and app_running:
        mins, secs = divmod(countdown_seconds, 60)
        countdown_label.configure(text=f"⏳ {mins:02}:{secs:02}")
        if countdown_seconds > 0:
            countdown_seconds -= 1
            tab1.after(1000, update_countdown)
        else:
            end_session()

def end_session():
    global app_running, vs, log_file, log_path
    app_running = False
    if vs:
        vs.release()
    if log_file:
        log_file.close()
    tabview.set("\ud83d\udcca View Current Result")
    view_results_from_csv(results_frame, log_path)
    export_pdf_btn.configure(command=lambda: export_figure_to_pdf(tab2, log_path))

    start_btn.pack(pady=10)
    timer_entry.pack()
    timer_label.pack()
    time_label.pack_forget()
    countdown_label.place_forget()

def pause_session():
    global paused, pause_start, start_time, paused_time
    if not app_running:
        return

    paused = not paused
    if paused:
        pause_start = time.time()
        pause_btn.configure(text="▶️ Resume Session", fg_color="#4CAF50", hover_color="#388E3C")
    else:
        paused_duration = time.time() - pause_start
        start_time += paused_duration  # Bù thời gian đã tạm dừng
        pause_btn.configure(text="⏸️ Pause Session", fg_color="#FFC107", hover_color="#FFA000")



def update_timer():
    if not app_running:
        return
    if paused:
        tab1.after(1000, update_timer)
        return

    elapsed = int(time.time() - start_time - 2)
    minutes = elapsed // 60
    seconds = elapsed % 60
    time_label.configure(text=f"🕒 {minutes:02d}:{seconds:02d}")
    tab1.after(1000, update_timer)


def update_video():
    global app_running, vs, frame_label, log_writer, drowsy_display_frames, yawn_display_frames, alarm_status, alarm_status2, alarm_status3, COUNTER, YAWN_COUNTER, FACE_MISSING_COUNTER

    if not app_running:
        return

    if paused:
        tab1.after(100, update_video)  # vẫn tiếp tục gọi lại nhưng không xử lý
        return

    ret, frame = vs.read()
    if not ret:
        return

    frame = cv2.resize(frame, (500, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if len(faces) == 0:
        FACE_MISSING_COUNTER += 1
        cv2.putText(frame, "FACE NOT DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if FACE_MISSING_COUNTER == FACE_MISSING_THRESHOLD and not alarm_status3:
            alarm_status3 = True
            Thread(target=sound3.play, daemon=True).start()
        log_writer.writerow([ts, "None", "-", "-", "No", "No", "Yes"])
    else:
        FACE_MISSING_COUNTER = 0
        alarm_status3 = False

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

# === MODERN UI ===
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Focus Tracker App")
# Lấy kích thước màn hình
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set kích thước cửa sổ gần full màn hình (trừ viền nhỏ)
root.geometry(f"{screen_width}x{screen_height}+0+0")

tabview = ctk.CTkTabview(root)
tabview.pack(expand=True, fill="both", padx=20, pady=20)

tab1 = tabview.add("\ud83d\udcf7 Start Session")
tab2 = tabview.add("\ud83d\udcca View Current Result")
tab4 = tabview.add("\ud83d\udcc1 Past Sessions")

# Tab 1
start_btn = ctk.CTkButton(tab1, text="▶️ Start Session", command=start_session, font=("Arial", 16))
start_btn.pack(pady=10)

stop_btn = ctk.CTkButton(tab1, text="⏹️ End Session", command=end_session, font=("Arial", 16), fg_color="#E53935", hover_color="#C62828")
stop_btn.pack(pady=10)

pause_btn = ctk.CTkButton(
    tab1, text="⏸️ Pause Session", command=pause_session,
    font=("Arial", 16), fg_color="#FFC107", hover_color="#FFA000"
)
pause_btn.pack(pady=10)

time_label = ctk.CTkLabel(tab1, text="🕒 00:00", font=("Arial", 16))
time_label.pack(pady=10)

timer_label = ctk.CTkLabel(tab1, text="⏱️ Thời gian (phút, để trống nếu không giới hạn):", font=("Arial", 10))
timer_label.pack(side='left', padx=10)

timer_entry = ttk.Entry(tab1, width=10)
timer_entry.pack(side='left')

countdown_label = ctk.CTkLabel(tab1, font=("Arial", 14), text_color="red")
countdown_label.place(relx=0.01, rely=0.95, anchor='sw')
countdown_label.place_forget()

frame_label = ctk.CTkLabel(tab1, text="", width=700, height=500)
frame_label.pack(pady=10)

# Tab 4
log_listbox = ctk.CTkOptionMenu(tab4, values=["No logs found"], command=lambda choice: on_log_select(choice))
log_listbox.pack(pady=20)

# Tab 2: Add Export PDF button and a frame for results
export_pdf_btn = ctk.CTkButton(tab2, text="📄 Xuất Kết Quả Ra PDF", font=("Arial", 16))
export_pdf_btn.pack(pady=10)



results_frame = ctk.CTkFrame(tab2)
results_frame.pack(expand=True, fill='both')

def on_log_select(choice):
    global log_path, results_frame

    log_path = os.path.join("logs", choice)

    # Nếu results_frame đã bị huỷ (do tab chưa hiển thị), tạo lại
    if not results_frame.winfo_exists():
        results_frame = ctk.CTkFrame(tab2)
        results_frame.pack(expand=True, fill='both')

    tabview.set("\ud83d\udcca View Current Result")
    view_results_from_csv(results_frame, log_path)
    export_pdf_btn.configure(command=lambda: export_figure_to_pdf(tab2, log_path))

def list_logs():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logs = sorted(os.listdir("logs"))
    if logs:
        log_listbox.configure(values=logs)
    else:
        log_listbox.configure(values=["No logs found"])

refresh_btn = ctk.CTkButton(tab4, text="🔄 Refresh Logs", command=list_logs)
refresh_btn.pack()

list_logs()

root.mainloop()
