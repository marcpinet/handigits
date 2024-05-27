import cv2
import numpy as np
import mediapipe as mp


class_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
input_shape = (64, 64)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)


def preprocess_image(hand_image, hand_landmarks):
    black_image = np.zeros_like(hand_image)
    mp_drawing.draw_landmarks(
        black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=None, connection_drawing_spec=connection_drawing_spec
    )
    
    gray = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
    gray[gray > 0] = 255
    
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        hand_contour = max(contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(hand_contour)
        
        zoom_factor = min(input_shape[0] / w, input_shape[1] / h)
        
        zoomed_size = (int(w * zoom_factor), int(h * zoom_factor))
        
        zoomed_hand = cv2.resize(gray[y:y+h, x:x+w], zoomed_size)
        
        zoomed_image = np.zeros(input_shape, dtype=np.uint8)
        
        y_offset = (input_shape[0] - zoomed_size[1]) // 2
        x_offset = (input_shape[1] - zoomed_size[0]) // 2
        
        zoomed_image[y_offset:y_offset+zoomed_size[1], x_offset:x_offset+zoomed_size[0]] = zoomed_hand
        
        resized = zoomed_image
    else:
        resized = cv2.resize(gray, input_shape)
    
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=-1)
    return np.expand_dims(reshaped, axis=0), resized


def segment_hand(image, scale_factor: float = None):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None, None, None
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
        return hand, hand_landmarks, (x_min, y_min, x_max, y_max)


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
                segmentation_result = segment_hand(frame, scale_factor)
                
                if segmentation_result[0] is not None:
                    segmented_hand, segmented_landmarks, (x_min, y_min, x_max, y_max) = segmentation_result
                    preprocessed_image, preprocessed_display = preprocess_image(segmented_hand, segmented_landmarks)
                    predictions = model.predict(preprocessed_image)
                    predicted_class = np.argmax(predictions)
                    predicted_label = class_names[predicted_class]
                    
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS, 
                        connection_drawing_spec=connection_drawing_spec
                    )
                    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    preprocessed_display = cv2.cvtColor(preprocessed_display, cv2.COLOR_GRAY2BGR)
                    frame[y_min:y_min+preprocessed_display.shape[0], x_min:x_min+preprocessed_display.shape[1]] = preprocessed_display
                else:
                    cv2.putText(frame, 'Hand not segmented', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'No hand detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Hand Sign Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
