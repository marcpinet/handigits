import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback


class HistoryLogger(Callback):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def on_epoch_end(self, epoch: int, logs: dict = None):
        if logs is not None:
            history = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in logs.items()}
            np.save(self.filepath, history)


def create_model(input_shape: tuple[int, int, int], num_classes: int) -> Sequential:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_callbacks(model_path: str, patience: int = 10) -> list[Callback]:
    return [
        EarlyStopping(patience=patience, restore_best_weights=True),
        ModelCheckpoint(f"{model_path}", save_best_only=True),
        HistoryLogger(model_path.replace('.keras', '_history.npy'))
    ]
