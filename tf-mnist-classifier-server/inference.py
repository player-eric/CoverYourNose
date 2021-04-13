import cv2 as cv
import numpy as np
import tensorflow as tf


def preprocess(image):
    image = cv.resize(image, (28, 28))
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image = 255 - image
    image /= 255
    image = tf.convert_to_tensor(image)
    return image


def inference(image):
    global model
    image = preprocess(image)
    digits = model.predict(image)
    prediction = np.argmax(digits)
    message = f"This is number {prediction}!"
    return message


def load_model(model_path):
    global model
    model = tf.keras.models.load_model(model_path)


def load_image(image_path):
    return cv.imread(image_path, cv.IMREAD_GRAYSCALE)
