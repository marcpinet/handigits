import cv2
import numpy as np
import mediapipe as mp
from neuralnetlib.models import Sequential
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


class_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)


def extract_landmarks(hand_landmarks: NormalizedLandmarkList) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()


def draw_landmarks_visualization(landmarks: np.ndarray, size: tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    Create a 2D visualization of hand landmarks with proper proportions
    """
    vis_img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Reshape landmarks back to (21, 3) for x, y, z coordinates
    landmarks_reshaped = landmarks.reshape(-1, 3)
    
    # Extract x and y coordinates
    x_coords = landmarks_reshaped[:, 0]
    y_coords = landmarks_reshaped[:, 1]
    
    # Calculate scaling while preserving aspect ratio
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    
    # Use the larger range to determine scaling to preserve aspect ratio
    scale = min(size[0] / max(x_range, y_range), size[1] / max(x_range, y_range)) * 0.8
    
    # Center the hand in the visualization
    x_mid = (x_coords.max() + x_coords.min()) / 2
    y_mid = (y_coords.max() + y_coords.min()) / 2
    
    x_normalized = ((x_coords - x_mid) * scale + size[0]/2).astype(int)
    y_normalized = ((y_coords - y_mid) * scale + size[1]/2).astype(int)
    
    # Draw connections (simplified hand skeleton)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
    ]
    
    # Draw connections first
    for start, end in connections:
        cv2.line(vis_img, 
                (x_normalized[start], y_normalized[start]),
                (x_normalized[end], y_normalized[end]),
                (0, 255, 0), 1)
    
    # Draw points on top
    for x, y in zip(x_normalized, y_normalized):
        cv2.circle(vis_img, (x, y), 1, (0, 255, 0), -1)
    
    return vis_img


def segment_hand(image: np.ndarray, scale_factor: float) -> tuple[np.ndarray, NormalizedLandmarkList, tuple[int, int, int, int]]:
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
        return hand, hand_landmarks, (x_min, y_min, x_max, y_max)


def start_live_recognition(model: Sequential):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    scale_factor = 1.5
    landmark_vis_size = (100, 100)  # Size of the landmark visualization
    
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
                    landmarks_vector = extract_landmarks(segmented_landmarks)
                    preprocessed_landmarks = np.expand_dims(landmarks_vector, axis=0)
                    predictions = model.predict(preprocessed_landmarks)
                    predicted_class = np.argmax(predictions)
                    predicted_label = class_names[predicted_class]
                    
                    # Draw the main hand landmarks and bounding box
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        connection_drawing_spec=connection_drawing_spec
                    )
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Create and add the landmark visualization
                    landmark_vis = draw_landmarks_visualization(landmarks_vector, landmark_vis_size)
                    # Place the visualization in the top-left corner of the bounding box
                    frame[y_min:y_min+landmark_vis_size[0], x_min:x_min+landmark_vis_size[1]] = \
                        cv2.addWeighted(
                            frame[y_min:y_min+landmark_vis_size[0], x_min:x_min+landmark_vis_size[1]],
                            0.2,  # Reduced transparency of the original frame
                            landmark_vis,
                            0.8,  # Increased visibility of the visualization
                            0
                        )
                    
                    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'Hand not segmented', (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'No hand detected', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Hand Sign Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()