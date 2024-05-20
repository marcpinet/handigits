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


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return True
    return False


def unzip_dataset(zip_path, output_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        root = zip_ref.namelist()[0].split("/")[0]
        for file in tqdm(zip_ref.namelist(), desc="Extracting files", unit="files"):
            if any(f"{root}/Dataset/{x}" in file for x in range(10)):
                zip_ref.extract(file, output_path)
                os.rename(output_path + file, output_path + '/'.join(file.split("/")[-2:]))
    os.remove(zip_path)
    shutil.rmtree(output_path + root + "/Dataset", ignore_errors=True)
    shutil.rmtree(output_path + root, ignore_errors=True)


def main():
    
    # Getting data if not exists
    if create_directory_if_not_exists(data_path):
        print("Downloading dataset to", raw_data_path, "...")
        create_directory_if_not_exists(raw_data_path)
        create_directory_if_not_exists(processed_data_path)
        download_file(dataset_url, raw_data_path + "dataset.zip")
        unzip_dataset(raw_data_path + "dataset.zip", raw_data_path)
        print("Dataset downloaded and extracted.")
    
        # Preprocessing data
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
    
        # Initializing model
        if create_directory_if_not_exists(model_path):
            print("Creating model...")
            input_shape = X_train.shape[1:]
            num_classes = y_train.shape[1]
            model = create_model(input_shape, num_classes)
            model.summary()
            print("Model created.")
            
            # Training model
            print("Training model...")
            model = create_model(input_shape, num_classes)
            callbacks = get_callbacks(model_path + "model.keras")
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, callbacks=callbacks)
            model.save(model_path + "final_model.keras")
            print("Model trained and saved.")
        
    # Inference on live video
    print("Starting live recognition...")
    model = tf.keras.models.load_model(model_path + "final_model.keras")
    start_live_recognition(model)


if __name__ == "__main__":
    main()
