import numpy as np
from keras.models import load_model
import keras_metrics
from sklearn import metrics
import glob2
import argparse
import cv2

def load_models(folder, streams):

    models_list = []

    for stream in streams:
        print('\nLoading', stream, 'model\n')
        models_list.append(load_model(folder + stream + '_best-weights.h5', custom_objects={'binary_precision':keras_metrics.binary_precision(), 'binary_recall':keras_metrics.binary_recall()}))

    return models_list

def load_data(folder, streams):

    classes = ['Falls', 'NotFalls']
    data = []
    labels = []

    for stream in streams:

        print('\nLoading', stream, 'data\n')
        imgs = []
        label = []
        for d_class in classes:

            file_list = glob2.glob(folder + d_class + '/*/' + stream + '*.npy')

            for file in file_list:

                img = np.load(file)
                imgs.append(img)

                if d_class == 'Falls':
                    label.append(0)
                else:
                    label.append(1)

        data.append(imgs)
        labels.append(label)

    data = np.asarray(data)
    labels = np.asarray(labels)
    labels = np.expand_dims(labels, axis=4)

    print(data.shape, labels.shape)
    return data, labels



def main(data_folder, model_folder, streams):

    models = load_models(model_folder, streams)

    data_raw, data_labels = load_data(data_folder, streams)
    print(data_raw[0].shape, data_labels[0].shape)

    predictions = models[0].predict(data_raw[0])
    predictions = np.argmax(predictions, axis=1)

    print(np.unique(predictions), predictions.shape)

    print(metrics.confusion_matrix(data_labels[0], predictions))
    print(metrics.classification_report(data_labels[0], predictions))
    print('Balanced Accuracy:', '{0:.2f}'.format(metrics.balanced_accuracy_score(data_labels[0], predictions)))
    print('Matthews:', '{0:.2f}'.format(metrics.matthews_corrcoef(data_labels[0], predictions)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Data help', required=True)
    parser.add_argument('--model', help='Model help', required=True)
    parser.add_argument('--streams', help='Streams help', nargs='+', required=True)

    args = parser.parse_args()

    data_folder = args.data
    model_folder = args.model
    streams = args.streams
    streams.sort()

    main(data_folder, model_folder, streams)
