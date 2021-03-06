import numpy as np
import tensorflow as tf
import os
import base64

from flask import Flask, request
from flask import current_app as app
from flask import render_template, jsonify
from PIL import Image
from inference import load_model, preprocess, load_image, inference
from io import BytesIO
from datetime import datetime

UPLOAD_FOLDER = './tmp'
app = Flask(__name__,
            instance_relative_config=False,
            template_folder="templates",
            static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home_endpoint():
    return render_template(
        'home.html')


@app.route('/predict', methods=['POST'])
def get_prediction():
    image = request.files['file']
    if image.filename != '':
        fn = os.path.join(
            app.config['UPLOAD_FOLDER'], image.filename +
            str(datetime.now().time())
        )
        image.save(fn)

        image = load_image(fn)
        res, preprocessed_image = inference(image)

        preprocessed_image = Image.fromarray(
            np.uint8(preprocessed_image * 255)).convert('RGB')
        buffer = BytesIO()
        preprocessed_image.save(buffer, format="PNG")
        myimage = buffer.getvalue()

        return jsonify(message=res, image=str(base64.b64encode(myimage))[2:-1])


if __name__ == '__main__':
    model = None
    load_model('model')
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
