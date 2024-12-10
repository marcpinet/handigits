import numpy as np
from neuralnetlib.models import Sequential
from neuralnetlib.layers import Dense, Dropout, BatchNormalization, Input
from neuralnetlib.callbacks import EarlyStopping, Callback


class HistoryLogger(Callback):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def on_epoch_end(self, epoch: int, logs: dict = None):
        if logs is not None:
            history = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in logs.items()}
            np.save(self.filepath, history)


def create_model(input_shape: int, num_classes: int) -> Sequential:
    model = Sequential()
    model.add(Input(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss_function='categorical_crossentropy')
    return model


def get_callbacks(model_path: str, patience: int = 10) -> list[Callback]:
    return [
        EarlyStopping(patience=patience, restore_best_weights=True),
        HistoryLogger(model_path.replace('.nnlb', '_history.npy'))
    ]
