"""Simple server to display code dependencies."""

import os
import uuid

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)

_models = []

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

@app.route("/processimage", methods=['POST'])
def processimage():
    """Returns the home page."""
    if request.method == 'POST':
        img = request.files["image"]
        if img:
            img_filename = os.path.join(os.getcwd() + "/static", str(uuid.uuid4()))
            img.save(img_filename)
            if not _models:
                models_dir = os.path.join("/vagrant", "fires", "new_model")
                for dirname, _, filenames in os.walk(models_dir):
                    for filename in filenames:
                        path = os.path.join(dirname, filename)
                        _models.append(load_model(path))

            img = image.load_img(img_filename)
            img = image.img_to_array(img) / 255
            img = tf.image.resize(img, (256, 256))
            img = tf.expand_dims(img, axis=0)

            predictions = []
            for model in _models:
                try:
                    predictions.append(
                        int(tf.round(model.predict(x=img)).numpy()[0][0])
                    )
                except Exception as ex:
                    print(ex)

            has_fire, does_not_have_fire = 0,0
            for p in predictions:
                if int(p) == 0:
                    has_fire += 1
                else:
                    does_not_have_fire += 1

            probability = has_fire * 1. /(has_fire + does_not_have_fire)

            return jsonify(
                {
                    "has_fire": probability > 0.5,
                    "predictions": [ "NO" if p == 1 else "YES"  for p in predictions]

                }
            )


    return "ok"


@app.route("/")
def home_route():
    """Returns the home page."""
    return render_template(
        'index.html'
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8889, debug=True)
