import os
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)


def preprocess_image(hand_image: np.ndarray, hand_landmarks: NormalizedLandmarkList) -> np.ndarray:
    black_image = np.zeros_like(hand_image)
    
    mp_drawing.draw_landmarks(
        black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=None, connection_drawing_spec=connection_drawing_spec
    )
    
    gray = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
    gray[gray > 0] = 255
    
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=-1)
    return reshaped


def load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        i = 1
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            for image_file in tqdm(os.listdir(label_dir), desc=f"Processing label {label} ({i}/{len(os.listdir(data_dir))})"):
                image_path = os.path.join(label_dir, image_file)
                image = cv2.imread(image_path)
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    preprocessed_image = preprocess_image(image, hand_landmarks)
                    images.append(preprocessed_image)
                    labels.append(int(label))
            i += 1
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def preprocess_data(images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = to_categorical(labels, num_classes=10)
    return train_test_split(images, labels, test_size=0.2, random_state=42)
