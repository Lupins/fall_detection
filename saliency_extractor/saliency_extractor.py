'''
'''
import tensorflow as tf
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import pickle
import os
import sys
import time
import cv2
slim=tf.contrib.slim

if not os.path.exists('models/research/slim'):
    return_code = subprocess.call(
        'git clone https://github.com/tensorflow/models',
        shell = True)
old_cwd = os.getcwd()
os.chdir('models/research/slim')
from nets import inception_v3
os.chdir(old_cwd)

# From our repository.
import saliency

# Boilerplate methods.
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

def LoadImage(file_path):
  #im = PIL.Image.open(file_path)
  im = cv2.imread(file_path)
  im = np.asarray(im)
  return im / 127.5 - 1.0

ckpt_file = './inception_v3.ckpt'

with tf.device('/cpu:0'):
    graph = tf.Graph()

    with graph.as_default():
      images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

      with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        _, end_points = inception_v3.inception_v3(images, is_training=False, num_classes=1001)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Restore the checkpoint
        sess = tf.Session(graph=graph, config=config)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_file)

      # Construct the scalar neuron tensor.
      logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
      neuron_selector = tf.placeholder(tf.int32)
      y = logits[0][neuron_selector]

      # Construct tensor for predictions.
      prediction = tf.argmax(logits, 1)

    # Load the image
    print(sys.argv[1])
    im = LoadImage(sys.argv[1])
    im = cv2.resize(im, (299, 299))

    start = time.time()

    # Make a prediction.
    prediction_class = sess.run(prediction, feed_dict = {images: [im]})[0]

    print("Prediction class: " + str(prediction_class))  # Should be a doberman, class idx = 237

    # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
    integrated_gradients = saliency.IntegratedGradients(graph, sess, y, images)

    # Baseline is a black image.
    baseline = np.zeros(im.shape)
    baseline.fill(-1)

    # Smoothed mask for integrated gradients will take a while since we are doing nsamples * nsamples computations.
    smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
      im, feed_dict = {neuron_selector: prediction_class}, x_steps=20, x_baseline=baseline)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)

    im = smoothgrad_mask_grayscale * 255
    im = cv2.resize(im, (224, 224))

    end = time.time()

    # Fetches only the number and extension of the file name
    file_name = sys.argv[1].split('_')[-1]
    file_name = 'saliency_' + file_name

    # Fetches everything in the path except the file name
    folder = sys.argv[1].split('/')[0:-1]
    # Concatenates the path back together
    path = ''
    for i in folder:
        path = path + i + '/'

    cv2.imwrite(path + file_name, im)
    print('TIME:', end-start)
