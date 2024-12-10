import os
import numpy as np
import cv2
import mediapipe as mp
from neuralnetlib.utils import train_test_split
from neuralnetlib.preprocessing import one_hot_encode
from tqdm import tqdm
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


mp_hands = mp.solutions.hands


def extract_landmarks(hand_landmarks: NormalizedLandmarkList) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()


def load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    landmarks = []
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
                    landmarks_vector = extract_landmarks(hand_landmarks)
                    landmarks.append(landmarks_vector)
                    labels.append(int(label))
            i += 1
    
    landmarks = np.array(landmarks)
    labels = np.array(labels)
    return landmarks, labels


def preprocess_data(landmarks: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = one_hot_encode(labels, num_classes=10)
    return train_test_split(landmarks, labels, test_size=0.2, random_state=42)
