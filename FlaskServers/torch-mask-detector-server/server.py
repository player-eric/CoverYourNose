import numpy as np
import os
import base64

from flask import Flask, request
from flask import current_app as app
from flask import render_template, jsonify
from PIL import Image
from inference import load_model,  load_image, run_on_image
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
        os.remove(fn)
        res_image, res = run_on_image(image)
        res_image = Image.fromarray(np.uint8(res_image)).convert('RGB')
        image_height_over_width = res_image.size[1] / res_image.size[0]
        #res_image = res_image.resize((260, int(image_height_over_width * 260)))
        buffer = BytesIO()
        res_image.save(buffer, format="PNG")
        return_image = buffer.getvalue()

        return jsonify(message=res, image=str(base64.b64encode(return_image))[2:-1])


if __name__ == '__main__':
    model = None
    load_model('mask_detector/models/model360.pth')
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
