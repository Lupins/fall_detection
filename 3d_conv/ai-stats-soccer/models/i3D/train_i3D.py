"""Script for training Inception-v1 Inflated 3D ConvNet"""

import os
import sys
sys.path.insert(0, "/home/leite/workspace/fall/3d_conv/ai-stats-soccer/models/i3D/")
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# THESE ARE SO THAT IT WILL USE ONLY THE CPU, SINCE I AM STILL DEBUGING THIS
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import imgaug.augmenters as iaa
import imgaug as ia
import glob2
from sklearn.model_selection import train_test_split
import keras_metrics
import hashlib

# from .soccer_functions import transform_data
from i3D_softmax import import_model


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataframe, stream, folder, batch_size, train_class=False, start_idx = 0, end_idx = 24, norm = 'div', file_prefix = '', load_before=False, aug=False, dim=(24, 224, 224, 3), n_classes=11, shuffle=False):
        'Initialization'
        self.dataframe = dataframe
        self.stream = stream
        self.folder = folder
        self.load_before = load_before
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.train_class = train_class
        self.norm = norm
        self.file_prefix = file_prefix
        self.filenames = pd.Series(glob2.glob(self.folder + '**/*.npy'))

        self.dataframe = self.dataframe[self.dataframe['stream'] == self.stream]

        if self.train_class != False:
            true_values = np.zeros(len(self.dataframe), dtype=bool)
            if type(self.train_class) == list:
                for label in self.train_class:
                    true_values = true_values | (dataframe['type'] == label)
            else:
                true_values = dataframe['type'] == self.train_class
            self.labels = pd.get_dummies(true_values)
        else:
            self.labels = pd.get_dummies(dataframe['class'])

        self.dim = dim
        self.aug = aug
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

        if self.load_before == True:
            self.X_rgb = np.empty(shape=(len(self.dataframe), *self.dim))
            self.y = np.empty(shape=(len(self.dataframe), self.n_classes))
            for i in range(len(self.dataframe)):
                video = str(self.dataframe.loc[ID].video)
                start = str(self.dataframe.loc[ID].start)
                end = str(self.dataframe.loc[ID].end)
                frame = np.load(self.folder + self.file_prefix + video + '_' + start + '_' + end + '.npy', allow_pickle=True).astype(float)[self.start_idx:self.end_idx]
                self.X_rgb[i] = frame
                self.y[i] = self.labels.loc[ID]
                if i%100 == 0:
                    print(i)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __apply_aug(self, X):
        aug_X = np.zeros_like(X)

        for i, batch in enumerate(X):
            augseq = iaa.Sequential([
                    iaa.Fliplr(0.5),
                    iaa.CropAndPad(
                        percent=(-0.08, 0.08),
                        pad_mode=["symmetric"]
                    ),
                    iaa.Add((-20, 20)),
                    iaa.AddToHueAndSaturation((-15, 15)),
                    iaa.Affine(
                        rotate=(-3, 3),
                        mode=["symmetric"]
                    )
                ])
            augseq_det = augseq.to_deterministic()
            aug_X[i] = np.array([augseq_det.augment_image(frame) for frame in batch])

        return aug_X

    def __getitem__(self, index):
        'Generate one batch of data'

#         Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         Generate data
        X, y = self.__data_generation(batch_indexes)

        if self.aug:
            X = self.__apply_aug(X.astype(np.uint8))

        if self.norm == 'div':
            X = X/255

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        'Generates data containing batch_size samples'

        if self.load_before:

            return self.X_rgb[batch_indexes], self.y[batch_indexes]
        else:


            X = np.empty(shape=(self.batch_size, *self.dim))
            y = np.empty(shape=(self.batch_size, self.n_classes))
            # list_IDs_temp = [self.dataframe.iloc[idx].video_name for idx in batch_indexes]
            list_IDs_temp = [self.dataframe.iloc[idx].name for idx in batch_indexes]

            for i, ID in enumerate(list_IDs_temp):

                filename = str(self.dataframe['path'].loc[ID])
                frame = np.load(filename, allow_pickle=True).astype(float)[self.start_idx:self.end_idx]

                # Store sample
                X[i] = frame

                # Store class
                y[i] = self.labels.loc[ID]

            return X, y


if __name__ == "__main__":

#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.log_device_placement = True

#     sess = tf.Session(config=config)
#     set_session(sess)

    batch_size = 8
    classes = 2
    # eval_class = ['shot', 'free-kick', 'goal', 'penalty-kick']
    eval_class = 'Falls'
    # dataset_folder = '/mnt/Data/leite/URFD/'
    dataset_folder = sys.argv[1]
    # current_stream = 'frame'
    current_stream = sys.argv[2]

    augmentation = None
    if sys.argv[3] == 'true':
        augmentation = True
    else:
        augmentation = False


    model = import_model(classes=classes, dim=(24,224,224,3), weight_name='rgb_imagenet_and_kinetics', dropout_prob=0.5)
#     model = import_model(classes=classes, dim=(24,224,224,3), weight_name='rgb_imagenet_and_kinetics')
    #model.load_weights('/home/ubuntu/soccer_videos/conv3d_model/saved_models/best-weights.h5')
#    parallel_model = multi_gpu_model(model, gpus=2)

    X_train = pd.read_csv(dataset_folder + 'train.csv')
    X_val = pd.read_csv(dataset_folder + 'validation.csv')
    X_test = pd.read_csv(dataset_folder + 'test.csv')

    # assert(hashlib.sha256(pd.util.hash_pandas_object(X_train, index=True).values).hexdigest() == '265f71d211c03c5203fa0a0f623d34a7e5242447937a712e1575afed75c69a1a')
    # assert(hashlib.sha256(pd.util.hash_pandas_object(X_val, index=True).values).hexdigest() == 'f23dec4b1a82799725b77d83e655c7e97aff4c28d27b095a69a0af6045147fc2')
    # assert(hashlib.sha256(pd.util.hash_pandas_object(X_test, index=True).values).hexdigest() == '8e10042ffc172e4112d19e7922d47e106892dd0afed24e7925cf54df41296637')

    train_df = X_train
    val_df = X_val

    train_rows = np.zeros(len(train_df), dtype=bool)
    if type(eval_class) == list:
        for label in eval_class:
            train_rows = train_rows | (train_df['type'] == label)
    else:
        train_rows = train_df['class'] == eval_class

#     Generators
    validation_generator = DataGenerator(val_df, current_stream, dataset_folder + 'validation/', batch_size=batch_size, n_classes=classes)
    training_generator = DataGenerator(train_df, current_stream, dataset_folder + 'train/', aug=augmentation, batch_size=batch_size, n_classes=classes)

    train_size = len(train_df)
    validation_size = len(val_df)

    y_true = np.array(pd.get_dummies(train_rows))

    false, true = np.sum(y_true, axis=0)
    total = false + true

    class_weight = {0: round(true/false, 2), 1: 1.}

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=10e-5, epsilon=1e-8), metrics=["categorical_accuracy", keras_metrics.binary_precision(), keras_metrics.binary_recall()])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    checkpoint = ModelCheckpoint('saved_models/' + current_stream + '_best-weights.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, es]

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=40,
                        steps_per_epoch=train_size//batch_size,
                        validation_steps=validation_size//batch_size,
                        callbacks=callbacks_list,
                        shuffle=False,
                        initial_epoch=0,
                        use_multiprocessing=True,
                        workers=3,
                        max_queue_size=10,
                        class_weight=class_weight)
