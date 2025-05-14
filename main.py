from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import imutils
import time
import dlib
import cv2
from pygame import mixer


mixer.init()
sound1 = mixer.Sound('wake_up.mp3')
sound2 = mixer.Sound('alert.mp3')


emotion_model = load_model('model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


alarm_status = False
alarm_status2 = False
saying = False

def alarm(msg):
    global alarm_status, alarm_status2, saying
    while alarm_status:
        print('Closed eyes alarm')
        saying = True
        sound1.play()
        saying = False
    if alarm_status2:
        print('Yawn alarm')
        sound2.play()


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


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


print("-> Khởi động webcam")
vs = VideoStream(src=0).start()
time.sleep(1.0)

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 35
YAWN_THRESH = 30
YAWN_CONSEC_FRAMES = 30
COUNTER = 0
YAWN_COUNTER = 0

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
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


        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)


        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = landmark_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, leftEye, rightEye = final_ear(shape)
        lip_dist = lip_distance(shape)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES and not alarm_status:
                alarm_status = True
                Thread(target=alarm, args=('wake up',), daemon=True).start()
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        if lip_dist > YAWN_THRESH:
            YAWN_COUNTER += 1
            if YAWN_COUNTER >= YAWN_CONSEC_FRAMES and not alarm_status2:
                alarm_status2 = True
                Thread(target=alarm, args=('take some fresh air',), daemon=True).start()
            cv2.putText(frame, "YAWN ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            YAWN_COUNTER = 0
            alarm_status2 = False


        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"YAWN: {lip_dist:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Emotion + Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
