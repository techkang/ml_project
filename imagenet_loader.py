"""Up to now, I only finished DNN part. So I need to convert the image to gray
image."""

import numpy as np
import os
import pickle

from imageio import imread

TRAIN_NUM = 400
EVALUATE_NUM = 490
TOTAL_NUM = 500


def create_data():
    """Load images from given dataset. The folder must be like
    Filename/1/image/image1.JPEG. After collect all data, save it to accelerate
    next load.

    """
    folders = os.listdir('dataset')
    data = np.ndarray((20, 500, 64, 64))
    for i, folder in enumerate(folders):
        images = os.listdir('dataset/' + folder + '/images')
        for j, image in enumerate(images):
            every = np.array(imread('dataset/' + folder + '/images/' + image))
            # print(i,j,every.shape)
            if len(every.shape) == 3:
                every = convert_to_grey(every)
            data[i][j] = every
    with open('imagenet.pkl', 'wb') as source:
        pickle.dump(data, source)


def load_data():
    """Load data from imagenet.pkl. Return a numpy.ndarray with shape (20, 500, 64, 64)."""
    with open('imagenet.pkl', 'rb') as source:
        row_data = pickle.load(source)
    train_data = [[], []]
    evaluate_data = [[], []]
    test_data = [[], []]
    for i, category in enumerate(row_data):
        train_data[0] += list(category[:TRAIN_NUM])
        train_data[1] += [i for _ in range(TRAIN_NUM)]
        evaluate_data[0] += list(category[TRAIN_NUM:EVALUATE_NUM])
        evaluate_data[1] += [i for _ in range(EVALUATE_NUM - TRAIN_NUM)]
        test_data[0] += list(category[EVALUATE_NUM:])
        test_data[1] += [i for _ in range(TOTAL_NUM - EVALUATE_NUM)]
    return train_data, evaluate_data, test_data


def convert_to_grey(image):
    """Convert a image to gray image. The image must be a numpy.ndarray. It
    will use function.
    We can use function Gray = R*0.299 + G*0.587 + B*0.114 to do this. By for
    accelerate calculation, we use function Gray =
    (R*30 + G*59 + B*11 + 50) / 100.

    """
    return image[:, :, 0] * 0.3 + image[:, :, 1] * 0.59 + image[:, :, 2] * 0.11


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 4096-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 4096-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (4096, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (4096, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (4096, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data


def vectorized_result(j):
    """Return a 20-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...19) into a corresponding desired output from the neural
    network."""
    e = np.zeros((20, 1))
    e[j] = 1
    return e


if __name__ == '__main__':
    load_data_wrapper()
