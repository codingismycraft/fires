"""Captures the camera and detects fire in the frame."""

import datetime
import os

import cv2

from keras.models import load_model
import tensorflow as tf


def detect_brightness(image, radius=71):
    """Paints a circle around the brighter part of the image.

    *******   CHANGES THE PASSED IN IMAGE BY REF ********
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    cv2.circle(image, maxLoc, 5, (255, 0, 0), 12)
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    cv2.circle(image, maxLoc, radius, (255, 0, 0), 2)


def load_models():
    models = []
    models_dir = os.path.join("new_model")
    for dirname, _, filenames in os.walk(models_dir):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            models.append(load_model(path))
    return models


def image_has_fire(rgb, models):
    img = rgb / 255
    img = tf.image.resize(img, (256, 256))
    img = tf.expand_dims(img, axis=0)
    predictions = []
    for model in models:
        try:
            predictions.append(
                int(tf.round(model.predict(x=img)).numpy()[0][0])
            )
        except Exception as ex:
            print(ex)
        if len(predictions) >= 3:
            break

    has_fire, does_not_have_fire = 0, 0
    for p in predictions:
        if int(p) == 0:
            has_fire += 1
        else:
            does_not_have_fire += 1
    print(has_fire, does_not_have_fire)
    probability = has_fire * 1. / (has_fire + does_not_have_fire)

    if probability >= 0.5:
        print("has fire")
    else:
        print("no fire")

    return probability >= 0.5


def main():
    last_time_spoke = datetime.datetime.now()
    models = load_models()

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)
    if not cap.isOpened():
        raise ConnectionError("Cannot open camera")

    frame_counter = 0
    has_fire = False
    while True:
        frame_counter += 1
        ret, frame = cap.read()

        if not ret:
            raise ConnectionError("Cannot read frame from camera.")

        if frame_counter % 10 == 0:
            # Check based on how many frames passed since last check.
            has_fire = image_has_fire(frame, models)
            frame_counter = 0

        image = cv2.cvtColor(frame, 0)

        if has_fire:
            cv2.putText(
                img=image,
                text="Fire",
                org=(60, 60),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=5
            )
            now = datetime.datetime.now()
            if (now - last_time_spoke).total_seconds() >= 1:
                os.system('spd-say "has fire.."')
                last_time_spoke = now
            detect_brightness(image)
        else:
            cv2.putText(
                img=image,
                text="no fire",
                org=(60, 60),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=2,
                color=(125, 246, 55),
                thickness=5
            )

        cv2.imshow('Sample Application.', image)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
