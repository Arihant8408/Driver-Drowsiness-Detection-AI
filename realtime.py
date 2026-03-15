import cv2
import numpy as np
from tensorflow.keras.models import load_model
import playsound
import threading

model = load_model("model.h5")

classes = ["closed", "normal"]

def play_alarm():
    playsound.playsound("alarm.wav")

cap = cv2.VideoCapture(0)

flag = 0
alarm_playing = False

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # resize frame smaller (less CPU)
    frame_small = cv2.resize(frame, (224,224))

    img = cv2.resize(frame_small, (64,64))
    img = img / 255.0
    img = np.reshape(img, (1,64,64,3))

    pred = model.predict(img, verbose=0)

    label = classes[np.argmax(pred)]

    cv2.putText(frame, label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 2)

    if label == "closed":
        flag += 1
    else:
        flag = 0
        alarm_playing = False

    # play alarm only once
    if flag > 5 and not alarm_playing:
        alarm_playing = True
        threading.Thread(
            target=play_alarm,
            daemon=True
        ).start()

    cv2.imshow("Drowsiness", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()