import numpy as np
import tensorflow as tf
import os
from flask import Flask, request
from inference import load_model, preprocess, load_image, inference

UPLOAD_FOLDER = './tmp'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home_endpoint():
    return 'I am an MNIST classifier!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    print("got a request")
    image = request.files['file']
    if image.filename != '':
        fn = os.path.join(
            app.config['UPLOAD_FOLDER'], image.filename
        )
        image.save(fn)

        image = load_image(fn)
        preprocess(image)
        res = inference(image)

        return res


if __name__ == '__main__':
    model = None
    load_model('model')
    app.run(host='127.0.0.1', port=1234)
