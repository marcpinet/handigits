import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback


class HistoryLogger(Callback):
    def __init__(self, filepath):
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            history = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in logs.items()}
            np.save(self.filepath, history)


def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_callbacks(model_path, patience=5):
    return [
        EarlyStopping(patience=patience, restore_best_weights=True),
        ModelCheckpoint(f"{model_path}", save_best_only=True),
        HistoryLogger(model_path.replace('.keras', '_history.npy'))
    ]