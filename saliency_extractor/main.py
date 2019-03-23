'''
    Saliency extractor

    Jump to main()
'''
import sys
import tensorflow as tf
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import pickle
import cv2
import saliency
from nets import inception_v3
slim = tf.contrib.slim

# IMPORTS ---------------------------------------------------------------------

import os
import subprocess
import tarfile

from six.moves import urllib

CKPT = None
PREDICTION = None
IMAGES = None

# GLOBAL VARIABLES ------------------------------------------------------------

INCEPTION_URL = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'

# SETUP FUNCTION --------------------------------------------------------------

def add_logit_tensor(ckpt_file):

    graph = tf.Graph()

    with graph.as_default():
        IMAGES = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            _, end_points = inception_v3.inception_v3(IMAGES, is_training=False, num_classes=1001)

            # Restore the checkpoint
            sess = tf.Session(graph=graph)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)

        # Construct the scalar neuron tensor
        logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
        neuron_selector = tf.placeholder(tf.int32)
        y = logits[0][neuron_selector]

        # Construct tensor for predictions
        prediction = tf.argmax(logits, 1)
        return prediction

# If not present, fetches inception model graph from INCEPTION_URL and
# extracts it to file: inception_v3.ckpt
def load_inception_model_graph(inception_url = INCEPTION_URL):

    # Check wether the model has already been downloaded
    if not os.path.exists('inception_v3.ckpt'):
        tgz_path = 'inception_v3_2016_08_28.tar.gz'

        # Download graph model from INCEPTION_URL
        print('Downloading inception model graph')
        urllib.request.urlretrieve(inception_url, tgz_path)
        print('Downloaded inception model graph')

        housing_tgz = tarfile.open(tgz_path) # Open downloaded file
        housing_tgz.extractall(path = './') # Extracts downloaded file
        housing_tgz.close() # Close file

    # Prepare return values
    ckpt_file = './inception_v3.ckpt'
    prediction = add_logit_tensor(ckpt_file)

    # return ckpt_file, prediction

# If not present, downloads tensorflow slim model from github source
def download_model():

    # Check wether the model has already been downloaded
    if not os.path.exists('models/research/slim'):
        # Clone git repository
        return_code = subprocess.call(
            'git clone https://github.com/tensorflow/models',
            shell = True)

    old_cwd = os.getcwd() # Save current directory
    os.chdir('models/research/slim/') # Move current directory
    from nets import inception_v3
    os.chdir(old_cwd) # move to previous saved directory
    print('Inception imported')

# Set proper values to global variables
def extractor_setup():
    download_model() # Download tensorflow slim model
    ckpt_file, prediction = load_inception_model_graph() # Download inception

    # Set global variables
    CKPT = ckpt_file
    PREDICTION = prediction

def saliency_extractor(frame, output_video):

    img = load_image(frame)
    prediction_class = sess.run(PREDICTION, feed_dict = {IMAGES: [img]})[0]

    integrated_smooth_gradient(img, prediction_class)

def integrated_smooth_gradient(img, pred_class):

    integrated_gradients = saliency.IntegratedGradients(graph, sess, y, images)

    baseline = np.zeros(img.shape)
    baseline.fill(-1)

    smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
        img, feed_dict = {neuron_selector: pred_class}, x_steps = 25, x_baseline=baseline)

    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)

    ROWS = 1
    COLS = 2
    UPSCALE_FACTOR = 10
    P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

    smoothgrad_mask_grayscale *= (255/smoothgrad_mask_grayscale.max())

    cv2.write('output.png', smoothgrad_mask_grayscale)

# -----------------------------------------------------------------------------

def iterate_over_video_frames(file):

    input_video = cv2.VideoCapture(file)
    output_video = None

    while(input_video.isOpened()):
        flag, frame = input_video.read()

        if flag:
            saliency_extractor(frame, output_video)

        else:
            break

    input_video.release()

def iterate_over_folder(path):

    file_list = os.listdir(path)

    for file_name in file_list:

        iterate_over_video_frames(file_name)

# BOILER PLATE FUNCTIONS ------------------------------------------------------

# Forgot why I did this
def load_image(frame):
    img = np.asarray(frame)

    return img / 127.5 - 1.0


# MAIN ------------------------------------------------------------------------
def main():

    # input_path = sys.args[1] # Path given by the user
    input_path = './' # TODO delete

    extractor_setup() # Prepare extractor
    iterate_over_folder(input_path) # Iterate over dataset

# Start here
main()
