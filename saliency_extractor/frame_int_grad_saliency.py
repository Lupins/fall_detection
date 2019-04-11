# Extracts saliency of an input image
import os
import sys
import time
import subprocess

import tensorflow as tf
import numpy as np
import cv2

import PIL.Image
from matplotlib import pylab as P
import pickle

import saliency

slim=tf.contrib.slim

OUT_RES_W = 224
OUT_RES_H = 224

IN_FOLDER = 'input/'

# Download inception model ----------------------------------------------------
if not os.path.exists('models/research/slim'):
    return_code = subprocess.call(
        'git clone https://github.com/tensorflow/models',
        shell = True)
old_cwd = os.getcwd()

# These two lines won't work outside of jupyter
os.chdir('models/research/slim')
from nets import inception_v3

os.chdir(old_cwd)

# -----------------------------------------------------------------------------

# Use either wget or curl depending on your OS.
if not os.path.exists('inception_v3.ckpt'):
  #!wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    return_code = subprocess.call(
        'curl -O http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
        shell = True)

    return_code = subprocess.call('tar -xvzf inception_v3_2016_08_28.tar.gz',
        shell = True)

ckpt_file = './inception_v3.ckpt'

GRAPH = tf.Graph()

# -----------------------------------------------------------------------------

def extract_vanilla(image, images, sess, logits, y, neuron_selector):
    return extract_saliency(image, 0, images, sess, logits, y, neuron_selector)

def extract_smooth(image, images, sess, logits, y, neuron_selector):
    return extract_saliency(image, 1, images, sess, logits, y, neuron_selector)

def extract_saliency(image, method, images, sess, logits, y, neuron_selector):

    prediction = tf.argmax(logits, 1)

    integrated_gradients = saliency.IntegratedGradients(GRAPH, sess, y, images)

    image = LoadImage(image)

    prediction_class = sess.run(prediction, feed_dict = {images:[image]})[0]

    baseline = np.zeros(image.shape)
    baseline.fill(-1)

    gradients_mask_3d = None
    # Vanilla
    if method == 0:
        gradients_mask_3d = integrated_gradients.GetMask(image,
                                                         feed_dict = {neuron_selector: prediction_class},
                                                         x_steps = 25,
                                                         x_baseline = baseline)

    # Smooth
    elif method == 1:
        gradients_mask_3d = integrated_gradients.GetSmoothedMask(image,
                                                                 feed_dict = {neuron_selector: prediction_class},
                                                                 x_steps = 25,
                                                                 x_baseline = baseline)

    grayscale_mask = saliency.VisualizeImageGrayscale(gradients_mask_3d)

    grayscale_mask *= (255 / gradients_mask_3d.max())
    grayscale_mask = np.uint8(grayscale_mask)

    return grayscale_mask

def extract_from_image(file_name, images, sess, logits, y, neuron_selector, out_w = OUT_RES_W, out_h = OUT_RES_H):

    # Open image file
    img_in = cv2.imread(IN_FOLDER + file_name)

    # Fetch video's information
    width, height, channels = img_in.shape

    # TODO Verify wether keep proportion is important
    img_in = resize_img(img_in, 299, 299)

    # n_frame = extract_vanilla(n_frame, images, sess, logits, y, neuron_selector)
    n_frame = extract_smooth(img_in, images, sess, logits, y, neuron_selector)

    # Setup output frame and write it to video file
    img_in = resize_img(img_in, out_w, out_h)

    # img_in = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)

    cv2.imwrite('output/' + file_name, img_in)

def main(file_name):

    print(file_name)
    start_time = time.time()

    with GRAPH.as_default():
        images = tf.placeholder(tf.float32, shape = (None, 299, 299, 3))

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            _, end_points = inception_v3.inception_v3(images,
                                                      is_training = False,
                                                      num_classes = 1001)

            # Restore the checkpoint
            sess = tf.Session(graph = GRAPH)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_file)

            # Construct the scalar neuron tensor
            logits = GRAPH.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
            neuron_selector = tf.placeholder(tf.int32)
            y = logits[0][neuron_selector]

            extract_from_image(file_name, images, sess, logits, y, neuron_selector)

            end_time = time.time()
            print(str(format(end_time - start_time, '.2f')), 's')

            os.remove('input/' + file_name)

# Boilerplate functions -------------------------------------------------------

def ShowImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')
  im = ((im + 1) * 127.5).astype(np.uint8)
  P.imshow(im)
  P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')

  P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
  P.title(title)

def ShowDivergingImage(grad, title='', percentile=99, ax=None):
  if ax is None:
    fig, ax = P.subplots()
  else:
    fig = ax.figure

  P.axis('off')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  im = ax.imshow(grad, cmap=P.cm.coolwarm, vmin=-1, vmax=1)
  fig.colorbar(im, cax=cax, orientation='vertical')
  P.title(title)

# Convert dtype uint8 to float64
def LoadImage(frame):
  im = np.asarray(frame)
  return im / 127.5 - 1.0

def urfd_crop_depth_info(frame, width, height):
    return frame[0:height, int(width / 2):width]

def resize_img(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

def remove_extension_name(name):
    name = name.split('.')[0]
    name = name.split('/')[1]
    return name
# -----------------------------------------------------------------------------

# iterate_over_folder('input')
main(str(sys.argv[1]))
