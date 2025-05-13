import cv2
import face_recognition
import mediapipe as mp
import numpy as np

from keras.models import load_model
from keras.utils import img_to_array

reference_image = face_recognition.load_image_file("my_photo.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)

name = "polina"
surname = "zimnyackova"

def detect_emotion(face_roi):
    if face_roi.size == 0:
        return "Neutral"
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    gray = gray.astype("float") / 255.0
    gray = img_to_array(gray)
    gray = np.expand_dims(gray, axis=0)

    preds = emotion_model.predict(gray)[0]
    emotion = emotion_labels[preds.argmax()]
    return emotion

def count_fingers(hand_landmarks, handedness):
    finger_count = 0
    tip_ids = [4, 8, 12, 16, 20]

    if (handedness == "Right" and hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x) or \
       (handedness == "Left" and hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x):
        finger_count += 1

    for tip in tip_ids[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_count += 1

    return finger_count

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_detection.process(rgb_frame)
    emotions = []
    is_recognized = False

    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)

            x = max(0, x)
            y = max(0, y)
            width = min(w - x, width)
            height = min(h - y, height)

            if width <= 0 or height <= 0:
                continue

            face_roi = frame[y:y+height, x:x+width]

            if face_roi.size > 0:
                try:
                    rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    face_encodings = face_recognition.face_encodings(rgb_face)
                    if face_encodings:
                        match = face_recognition.compare_faces([reference_encoding], face_encodings[0])
                        color = (0, 255, 0) if match[0] else (0, 0, 255)
                        is_recognized = match[0]
                    else:
                        color = (255, 255, 0)
                except:
                    color = (255, 255, 0)

                emotion = detect_emotion(face_roi)
                emotions.append(emotion)

                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

    hand_results = hands.process(rgb_frame)
    display_text = None

    if hand_results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            hand_label = handedness.classification[0].label
            fingers = count_fingers(hand_landmarks, hand_label)

            if is_recognized:
                if fingers == 1:
                    display_text = name
                elif fingers == 2:
                    display_text = f"{name} {surname}"
                elif fingers == 3 and emotions:
                    display_text = f"Emotion: {emotions[0]}"
            else:
                display_text = "Unknown"

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if display_text:
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion & Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()