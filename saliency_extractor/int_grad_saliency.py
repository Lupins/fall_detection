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

def extract_from_video(file_name, images, sess, logits, y, neuron_selector, out_w = OUT_RES_W, out_h = OUT_RES_H):

    # Open video file
    v_in = cv2.VideoCapture('input/' + file_name)

    # Fetch video's information
    width = int(v_in.get(3))
    height = int(v_in.get(4))
    n_frames = int(v_in.get(7))

    # Setup video output
    file_out = remove_extension_name(file_name)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    v_out = cv2.VideoWriter('output/' + file_out + '.avi', fourcc, v_in.get(5),
                            (out_w, out_h))

    # Iterate over video frames
    i = 0
    while(v_in.isOpened()):

        # Read next frame
        flag, frame = v_in.read()

        # if i == 30:
            # break

        # Was frame read correctly?
        if flag:

            # Report and keep track of progress
            print(format(i, '4d'), '|', format(n_frames, '4d'))
            i += 1

            # TODO remove this, prepare dataset before this program
            n_frame = urfd_crop_depth_info(frame, width, height)
            # TODO Verify wether keep proportion is important
            n_frame = resize_frame(n_frame, 299, 299)

            # Extract saliency and measure how long it took, in seconds
            start_time = time.time()
            n_frame = extract_vanilla(n_frame, images, sess, logits, y, neuron_selector)
            # n_frame = extract_smooth(n_frame, images, sess, logits, y)
            end_time = time.time()

            print(str(format(end_time - start_time, '.2f')), 's')

            # Setup output frame and write it to video file
            n_frame = resize_frame(n_frame, out_w, out_h)

            n_frame = cv2.cvtColor(n_frame, cv2.COLOR_GRAY2BGR)

            v_out.write(n_frame)

        else:
            break

    v_in.release()
    v_out.release()

def iterate_over_folder(path):

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

            # for file in os.listdir(path):

                # extract_from_video(file, images, sess, logits, y, neuron_selector)
                # os.remove('input/' + file) # Delete file after it's been extracted

            extract_from_video(path, images, sess, logits, y, neuron_selector)
            # os.remove('input/' + path)

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

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

def remove_extension_name(name):
    return name.split('.')[0]
# -----------------------------------------------------------------------------

# iterate_over_folder('input')
iterate_over_folder(sys.argv[1])
