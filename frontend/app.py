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

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class ActiveModels:
    """Holds the models that will be used to detect fires."""

    # The name of the pretrained models to use.
    _MODELS = [
        "model.h5",
        "model2.h5",
        "model3.h5",
        "model-11.h5",
        "model-12.h5"
    ]

    # Holds the model instances.
    _models = None

    @classmethod
    def get_models(cls):
        """Returns the models to use for fire detection.

        :return: A list of models.
        :rtype: list
        """
        if not cls._models:
            cls._load_models()
        return cls._models

    @classmethod
    def _load_models(cls):
        """Loads the models that will be used to detect fire.

        :return: A list of models.
        :rtype: list
        """
        cls._models = []
        models_dir = os.path.join(_CURRENT_DIR, "..", "fires", "new_model")
        for model_file_name in cls._MODELS:
            full_path = os.path.join(models_dir, model_file_name)
            cls._models.append(load_model(full_path))


@app.route("/processimage", methods=['POST'])
def processimage():
    """Returns the home page."""
    if request.method == 'POST':
        img = request.files["image"]
        if img:
            img_filename = os.path.join(os.getcwd() + "/static",
                                        str(uuid.uuid4()))
            img.save(img_filename)

            img = image.load_img(img_filename)
            img = image.img_to_array(img) / 255
            img = tf.image.resize(img, (256, 256))
            img = tf.expand_dims(img, axis=0)

            predictions = []
            for model in ActiveModels.get_models():
                try:
                    predictions.append(
                        int(tf.round(model.predict(x=img)).numpy()[0][0])
                    )
                except Exception as ex:
                    print(ex)

            has_fire, does_not_have_fire = 0, 0
            for p in predictions:
                if int(p) == 0:
                    has_fire += 1
                else:
                    does_not_have_fire += 1

            probability = has_fire * 1. / (has_fire + does_not_have_fire)

            return jsonify(
                {
                    "has_fire": probability > 0.5,
                    "predictions": ["NO" if p == 1 else "YES" for p in
                                    predictions]

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
