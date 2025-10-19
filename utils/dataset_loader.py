import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder_path, target_size=(224, 224)):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
    return np.array(images)

def load_dataset(data_dir, labels_dict, test_size=0.2):
    X, y = [], []
    for label, folder in labels_dict.items():
        folder_path = os.path.join(data_dir, folder)
        imgs = load_images_from_folder(folder_path)
        X.extend(imgs)
        y.extend([label] * len(imgs))
    return train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=42)
