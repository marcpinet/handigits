import os
import zipfile
import shutil
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tqdm import tqdm
from utils.retriever import download_file
from utils.preprocess import load_data, preprocess_data
from utils.model import create_model, get_callbacks
from utils.inference import start_live_recognition

data_path = "data/"
raw_data_path = data_path + "raw/"
processed_data_path = data_path + "processed/"
model_path = data_path + "model/"
dataset_url = "https://codeload.github.com/ardamavi/Sign-Language-Digits-Dataset/zip/refs/heads/master"


def create_directory_if_not_exists(directory: str) -> bool:
    if not os.path.exists(directory):
        os.makedirs(directory)
        return True
    return False


def unzip_dataset(zip_path: str, output_path: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        root = zip_ref.namelist()[0].split("/")[0]
        for file in tqdm(zip_ref.namelist(), desc="Extracting files", unit="files"):
            if any(f"{root}/Dataset/{x}" in file for x in range(10)):
                zip_ref.extract(file, output_path)
                os.rename(output_path + file, output_path + '/'.join(file.split("/")[-2:]))
    os.remove(zip_path)
    shutil.rmtree(output_path + root + "/Dataset", ignore_errors=True)
    shutil.rmtree(output_path + root, ignore_errors=True)


def directory_has_files(directory: str) -> bool:
    for _, _, files in os.walk(directory):
        if any(files):
            return True
    return False


def main():
    # Getting data
    if not os.path.exists(data_path) or not directory_has_files(data_path):
        create_directory_if_not_exists(data_path)
    if not os.path.exists(raw_data_path) or not directory_has_files(raw_data_path):
        print("Downloading dataset to", raw_data_path, "...")
        create_directory_if_not_exists(raw_data_path)
        download_file(dataset_url, os.path.join(raw_data_path, "dataset.zip"))
        unzip_dataset(os.path.join(raw_data_path, "dataset.zip"), raw_data_path)
        print("Dataset downloaded and extracted.")
    else:
        print("Raw data already exists. Skipping download.")

    # Preprocessing data
    if not os.path.exists(processed_data_path) or not directory_has_files(processed_data_path):
        print("Preprocessing data...")
        os.makedirs(processed_data_path, exist_ok=True)
        os.makedirs(os.path.join(processed_data_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(processed_data_path, 'test'), exist_ok=True)
        images, labels = load_data(raw_data_path)
        X_train, X_test, y_train, y_test = preprocess_data(images, labels)
        np.save(os.path.join(processed_data_path, 'train', 'X_train.npy'), X_train)
        np.save(os.path.join(processed_data_path, 'train', 'y_train.npy'), y_train)
        np.save(os.path.join(processed_data_path, 'test', 'X_test.npy'), X_test)
        np.save(os.path.join(processed_data_path, 'test', 'y_test.npy'), y_test)
        print("Data preprocessed and saved.")
    else:
        print("Processed data already exists. Skipping preprocessing.")

    X_train = np.load(os.path.join(processed_data_path, 'train', 'X_train.npy'))
    y_train = np.load(os.path.join(processed_data_path, 'train', 'y_train.npy'))
    X_test = np.load(os.path.join(processed_data_path, 'test', 'X_test.npy'))
    y_test = np.load(os.path.join(processed_data_path, 'test', 'y_test.npy'))

    # Initializing model
    if not os.path.exists(model_path) or not directory_has_files(model_path):
        create_directory_if_not_exists(model_path)
        print("Creating model...")
        input_shape = X_train.shape[1]
        num_classes = y_train.shape[1]
        model = create_model(input_shape, num_classes)
        model.summary()
        print("Model created.")

    # Training model
    print("Training model...")
    if not os.path.exists(os.path.join(model_path, "model.keras")):
        callbacks = get_callbacks(os.path.join(model_path, "model.keras"))
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=32, callbacks=callbacks)
        model.save(os.path.join(model_path, "final_model.keras"))
        print("Model trained and saved.")
    else:
        print("Model already trained. Skipping training.")
    
    # Inference on live video
    print("Starting live recognition...")
    model = tf.keras.models.load_model(os.path.join(model_path, "final_model.keras"))
    start_live_recognition(model)


if __name__ == "__main__":
    main()
