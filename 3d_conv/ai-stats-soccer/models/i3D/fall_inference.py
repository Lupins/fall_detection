from keras.models import load_model
from keras import metrics
import glob2
import argparse

def load_models(folder, streams):

    models_list = []
    
    for stream in streams:
        models_list.append(load_model(folder + stream + '_best-weights.h5', custom_objects={'binary_precision':keras_metrics.binary_precision(), 'binary_recall':keras_metrics.binary_recall()}))

    return models_list

def main(data_folder, model_folder, streams):

    models = load_models(model_folder, streams)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Data help', required=True)
    parser.add_argument('--model', help='Model help', required=True)
    parser.add_argument('--streams', nargs='+', required=True)

    args = parser.parse_args()

    data_folder = args.data
    model_folder = args.model
    streams = args.streams

    main(data_folder, model_folder, streams)
