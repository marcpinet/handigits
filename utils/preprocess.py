import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64))
            images.append(image)
            labels.append(int(label))
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def preprocess_data(images, labels):
    images = images / 255.0
    images = images[..., np.newaxis]
    labels = to_categorical(labels, num_classes=10)
    return train_test_split(images, labels, test_size=0.2, random_state=42)