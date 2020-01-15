import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image

'''def load_data(data_dir, dataset, data_size, phase, valid_split=None, seed=None):
    if phase == 'test':
        test_files = os.listdir(os.path.join(data_dir, dataset, 'test'))
        X_test, y_test = np.zeros((len(test_files), data_size, data_size, 3)), np.zeros(len(test_files))
        for i, f in enumerate(test_files):
            img = adjust_image_shape(Image.open(os.path.join(data_dir, dataset, 'test', f)), data_size)
            X_test[i, :, :, :] = np.asarray(img)
            y_test[i] = int((f.split('_')[0]))
        return X_test, y_test

    elif phase == 'train_valid':
        train_files = os.listdir(os.path.join(data_dir, dataset, 'train'))
        X_train, y_train = np.zeros((len(train_files), data_size, data_size, 3)), np.zeros(len(train_files))
        for i, f in enumerate(train_files):
            img = adjust_image_shape(Image.open(os.path.join(data_dir, dataset, 'train', f)), data_size)
            X_train[i, :, :, :] = np.asarray(img)
            y_train[i] = int((f.split('_')[0]))
        X_train, y_train = shuffle(X_train, y_train, random_state=seed)

        if os.path.isdir(os.path.join(data_dir, dataset, 'valid')):
            valid_files = os.listdir(os.path.join(data_dir, dataset, 'valid'))
            X_valid, y_valid = np.zeros((len(valid_files), data_size, data_size, 3)), np.zeros(len(valid_files))
            for i, f in enumerate(valid_files):
                img = adjust_image_shape(Image.open(os.path.join(data_dir, dataset, 'valid', f)), data_size)
                X_valid[i, :, :, :] = np.asarray(img)
                y_valid[i] = int((f.split('_')[0]))
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                                  test_size=valid_split,
                                                                  random_state=seed)
        return X_train, y_train, X_valid, y_valid

    else:
        raise ValueError('value of PHASE is not admitted')'''


def load_data(data_dir, dataset, data_size, valid_split, test_split, seed):
    files = os.listdir(os.path.join(data_dir, dataset))
    X, y = np.zeros((len(files), data_size, data_size, 3)), np.zeros(len(files))
    for i, f in enumerate(files):
        img = adjust_image_shape(Image.open(os.path.join(data_dir, dataset, f)), data_size)
        X[i, :, :, :] = np.asarray(img)
        y[i] = int((f.split('_')[0]))
    X, y = shuffle(X, y, random_state=seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_split, random_state=seed)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def adjust_image_shape(img, data_size):
    img = np.asarray(img)

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    if img.shape[-1] == 4:
        new_img = img[:, :, :3]
    else:
        new_img = img

    new_img = Image.fromarray(new_img)
    width, height = new_img.size
    if height >= data_size and width >= data_size:
        resized_img = new_img.resize((data_size, data_size))
    else:
        resized_img = Image.new("RGB", (data_size, data_size))
        resized_img.paste(new_img, (int((data_size - width) / 2), int((data_size - height) / 2)))

    return resized_img
