import joblib
import pandas as pd
import numpy as np
import mediapipe as mp
import cv2
import statistics
import imageio
from collections import deque
import time


model = joblib.load('chekpoints/xgb_model.pkl')
label_encoder = joblib.load('chekpoints/label_encoder.pkl')



def preprocess_landmarks(landmarks):
    coords = np.array(landmarks).reshape(-1, 3)
    wrist = coords[0][:2]
    mid_tip = coords[12][:2]
    scale = np.linalg.norm(mid_tip - wrist)
    coords[:, :2] = (coords[:, :2] - wrist) / (scale + 1e-6)
    return coords.flatten()

    
    
# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

pred_buffer = deque(maxlen=12)  # window size
frames = []
# Open webcam
cap = cv2.VideoCapture("http://192.168.1.3:4747/video")
cap = cv2.VideoCapture(0)

# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Flip and convert to RGB
#         image = cv2.flip(frame, 1)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         result = hands.process(image_rgb)
#         prediction = ""
#         if result.multi_hand_landmarks:
#             for hand_landmarks in result.multi_hand_landmarks:
#                 landmarks = []
#                 for lm in hand_landmarks.landmark:
#                     h, w, _ = image.shape
#                     landmarks.append([lm.x * w, lm.y * h, lm.z])

#                 processed = preprocess_landmarks(landmarks)
#                 # Remove Zs components
#                 processed = processed.reshape(-1,3)[:,:2].flatten()
#                 # Remove wrist point
#                 processed = processed[2:].reshape(1, -1)
#                 prediction = model.predict(processed)

#                 # Draw landmarks & prediction
#                 mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                 prediction = label_encoder.inverse_transform(prediction)[0]
#                 pred_buffer.append(prediction)
#                 prediction = statistics.mode(pred_buffer)

#                 cv2.putText(image, f'Gesture: {prediction}', (10, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                

#         cv2.imshow('Hand Gesture Prediction', image)
#         frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert to RGB
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = hands.process(image_rgb)
        prediction = ""

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, _ = image.shape
                    landmarks.append([lm.x * w, lm.y * h, lm.z])

                processed = preprocess_landmarks(landmarks)
                # Remove Zs components
                processed = processed.reshape(-1, 3)[:, :2].flatten()
                # Remove wrist point
                processed = processed[2:].reshape(1, -1)
                prediction = model.predict(processed)

                # Draw landmarks & prediction
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                prediction = label_encoder.inverse_transform(prediction)[0]
                pred_buffer.append(prediction)
                prediction = statistics.mode(pred_buffer)

                cv2.putText(image, f'Gesture: {prediction}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow('Hand Gesture Prediction', image)
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Smooth exit on 'ok'
        if prediction == "ok":
            print("👌 'ok' detected — fading out...")

            # Show label one last time
            cv2.putText(image, "Detected: OK", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow('Hand Gesture Prediction', image)
            cv2.waitKey(1)

            # Fade out
            import numpy as np
            for alpha in np.linspace(1, 0, 30):
                faded = (image * alpha).astype(np.uint8)
                frames.append(cv2.cvtColor(faded, cv2.COLOR_BGR2RGB))
                cv2.imshow("Hand Gesture Prediction", faded)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            break  # exit loop after fadeout

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    gif_path = "misc/gesture_demo.gif"
    imageio.mimsave(gif_path, frames, fps=12)

