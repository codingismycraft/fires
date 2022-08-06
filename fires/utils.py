"""Utility functions to handle the training data."""

import cv2
import datetime
import random
import enum
import os

import keras.utils.np_utils as np_utils
import matplotlib.pyplot as plt
import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# The directory that contains training non fire images.
_TRAINING_NON_FIRE_DIR = os.path.join(
    _CURRENT_DIR, "..", "data", "train-kaggle", "non_fire")

# The directory that contains training fire images.
_TRAINING_FIRE_DIR = os.path.join(
    _CURRENT_DIR, "..", "data", "train-kaggle", "fire"
)

# The directory that contains testing non fire images.
_TESTING_NON_FIRE_DIR = os.path.join(
    _CURRENT_DIR, "..", "data", "mytesting", "non_fire")

# The directory that contains testing fire images.
_TESTING_FIRE_DIR = os.path.join(
    _CURRENT_DIR, "..", "data", "mytesting", "fire"
)

# The number of classes we are looking for are 2 (fire / not fire)
NUMBER_OF_CLASSES = 2


class DataType(enum.Enum):
    """Designates the type of the data to load."""
    TRAINING = 1
    TESTING = 2


_DATA_DIRS = {
    DataType.TRAINING: [_TRAINING_NON_FIRE_DIR, _TRAINING_FIRE_DIR],
    DataType.TESTING: [_TESTING_NON_FIRE_DIR, _TESTING_FIRE_DIR]
}

def get_testing_data_files_with_fire(count=10):
    non_fire_dir, fire_dir = _DATA_DIRS[DataType.TESTING]
    filenames = []
    for index, filename in enumerate(os.listdir(fire_dir)):
        filenames.append(os.path.join(fire_dir, filename))
    random.shuffle(filenames)
    return filenames[:count]

def get_testing_data_files_with_no_fire(count=10):
    non_fire_dir, fire_dir = _DATA_DIRS[DataType.TESTING]
    filenames = []
    for index, filename in enumerate(os.listdir(non_fire_dir)):
        filenames.append(os.path.join(non_fire_dir, filename))
    random.shuffle(filenames)
    return filenames[:count]
    
def load(data_type, max_count=10, resolution=128, normalize=True):
    """Loads the data from the disk.

    Loads both fire and non-fire images and returns the images and the
    Corresponding labels (as a list of 2 dimensional lists like [0. 1]
    expressing the target for the image.

    :param DataType data_type: The type of the data to use (train or test).

    :param int max_count: The maximum number of images per category; the
    same count applies to both categories (meaning that we cannot specify
    a different maximum count for training and testing).

    :returns: A tuple consisting of the images and the labels.
    :rtype: tuple
    """
    images = []
    labels = []
    non_fire_dir, fire_dir = _DATA_DIRS[data_type]
    for label, parent_dir in enumerate([non_fire_dir, fire_dir]):
        for index, filename in enumerate(os.listdir(parent_dir)):
            if index >= max_count:
                break
            full_path = os.path.join(parent_dir, filename)
            if full_path.endswith("png") or full_path.endswith("jpg"):
                try:
                    img = cv2.resize(cv2.imread(full_path),
                                     (resolution, resolution))
                except Exception as ex:
                    print(str(ex))
                else:
                    images.append(img)
                    labels.append(label)

    if normalize:
        images = np.array(images, dtype='float32') / 255.
    labels = np_utils.to_categorical(
        np.array(labels), num_classes=NUMBER_OF_CLASSES
    )

    return images, labels


def display_image(img):
    """Displays the image based on the passed in full path.

    :param img: The image to display.
    """
    plt.imshow(img)
    plt.show()


def make_directory_name(score):
    """Makes a directory name where a model will be saved.

    :param float score: The score of the model (should be 0 -1).

    return: The name of the directory where to store the model. The
    directory name consists of the date - time and the score, for
    example something like this: 2022-08-01:17:10:55:0.8
    :rtype str.
    """
    assert 0. <= score <= 1.0
    now = datetime.datetime.now()
    x = str(score)[0:4]
    t = now.strftime('%Y-%m-%d:%H:%M:%S')
    dirname = f'{t}'
    return os.path.join(_CURRENT_DIR, "..", "models", f"{dirname}:{x}")


def get_all_saved_models():
    path = os.path.join(_CURRENT_DIR, "..", "models")
    return [os.path.join(path, root, "model.h5") for root in os.listdir(path)]


if __name__ == '__main__':
    get_testing_data_files_with_no_fire()

