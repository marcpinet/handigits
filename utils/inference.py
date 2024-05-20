import cv2
import numpy as np
import mediapipe as mp


class_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
input_shape = (64, 64)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    resized = cv2.resize(equalized, input_shape)
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=-1)
    return np.expand_dims(reshaped, axis=0)


def segment_hand(image, scale_factor):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None
        hand_landmarks = results.multi_hand_landmarks[0]
        x_min, x_max = min(lm.x for lm in hand_landmarks.landmark), max(lm.x for lm in hand_landmarks.landmark)
        y_min, y_max = min(lm.y for lm in hand_landmarks.landmark), max(lm.y for lm in hand_landmarks.landmark)
        x_min, x_max = int(x_min * image.shape[1]), int(x_max * image.shape[1])
        y_min, y_max = int(y_min * image.shape[0]), int(y_max * image.shape[0])
        size = max(x_max - x_min, y_max - y_min)
        size = int(size * scale_factor)
        x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2
        x_min, y_min = x_center - size // 2, y_center - size // 2
        x_max, y_max = x_min + size, y_min + size
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)
        hand = image[y_min:y_max, x_min:x_max]
        hand = cv2.resize(hand, input_shape)
        return hand, (x_min, y_min, x_max, y_max)


def start_live_recognition(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    scale_factor = 1.5
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                segmented_hand = segment_hand(frame, scale_factor)
                
                if segmented_hand is not None:
                    hand, (x_min, y_min, x_max, y_max) = segmented_hand
                    preprocessed_image = preprocess_image(hand)
                    predictions = model.predict(preprocessed_image)
                    predicted_class = np.argmax(predictions)
                    predicted_label = class_names[predicted_class]
                    
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Hand not segmented', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'No hand detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Hand Sign Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()