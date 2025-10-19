import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def normalize_image(img):
    img = img.astype(np.float32)
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def augment_image(img):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    img = np.expand_dims(img, axis=0)
    it = datagen.flow(img, batch_size=1)
    return next(it)[0]
