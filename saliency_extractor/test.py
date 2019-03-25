# Boilerplate imports.
import tensorflow as tf
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import pickle
import os
import cv2
import subprocess
import time
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

# %matplotlib inline

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

def LoadImage(frame):
#   im = PIL.Image.open(file_path)
#   im = np.asarray(im)
  im = np.asarray(frame)
  return im / 127.5 - 1.0

# Use either wget or curl depending on your OS.
if not os.path.exists('inception_v3.ckpt'):
  #!wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    return_code = subprocess.call(
        'curl -O http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
        shell = True)

    return_code = subprocess.call('tar -xvzf inception_v3_2016_08_28.tar.gz',
        shell = True)

ckpt_file = './inception_v3.ckpt'

graph = tf.Graph()

with graph.as_default():
  images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    _, end_points = inception_v3.inception_v3(images, is_training=False, num_classes=1001)

    # Restore the checkpoint
    sess = tf.Session(graph=graph)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

  # Construct the scalar neuron tensor.
  logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
  neuron_selector = tf.placeholder(tf.int32)
  y = logits[0][neuron_selector]

  # Construct tensor for predictions.
  prediction = tf.argmax(logits, 1)



def urfd_crop_depth_info(frame, width, height):
    return frame[0:height, int(width / 2):width]

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

def extract_saliency(frame, i):

    frame = LoadImage(frame)
    prediction_class = sess.run(prediction, feed_dict = {images: [frame]})[0]

    print("Prediction class: " + str(prediction_class))  # Should be a doberman, class idx = 237

# Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
    integrated_gradients = saliency.IntegratedGradients(graph, sess, y, images)

# Baseline is a black image.
    baseline = np.zeros(frame.shape)
    baseline.fill(-1)

# Compute the vanilla mask and the smoothed mask.
# vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
#   im, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
# Smoothed mask for integrated gradients will take a while since we are doing nsamples * nsamples computations.
    smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
      frame, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)

# Call the visualization methods to convert the 3D tensors to 2D grayscale.
# vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)

# Set up matplot lib figures.
    ROWS = 1
    COLS = 2
    UPSCALE_FACTOR = 10
    P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Render the saliency masks.
# ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Integrated Gradients', ax=P.subplot(ROWS, COLS, 1))
    ShowGrayscaleImage(smoothgrad_mask_grayscale, title='Smoothgrad Integrated Gradients', ax=P.subplot(ROWS, COLS, 2))

# vanilla_mask_grayscale *= (255 / vanilla_mask_grayscale.max())
    smoothgrad_mask_grayscale *= (255 / smoothgrad_mask_grayscale.max())

# temp_con = np.concatenate((vanilla_mask_grayscale, smoothgrad_mask_grayscale), axis=1)
# temp_con = np.concatenate((temp_im, temp_con), axis=1)
# result_con = np.concatenate((result_con, temp_con), axis=0)

    # print('Shape: ' + str(smoothgrad_mask_grayscale.shape) + ' ' + str(smoothgrad_mask_grayscale.dtype))
    smoothgrad_mask_grayscale = np.uint8(smoothgrad_mask_grayscale)
    # print('Shape: ' + str(smoothgrad_mask_grayscale.shape) + ' ' + str(smoothgrad_mask_grayscale.dtype))
    # cv2.imwrite('Output_' + str(i) + '.png', smoothgrad_mask_grayscale)

    return smoothgrad_mask_grayscale
    # cv2.imwrite('Output.png', smoothgrad_mask_grayscale)

def extract_from_video(file_name):

    video = cv2.VideoCapture('input/' + file_name)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')

# print('Video 5 '+ str(video.get(5)))
# print('3 and 4 ' + str(video.get(3)) + ' ' + str(video.get(4)))
    width = int(video.get(3))
    height = int(video.get(4))
# print('half: ' + str(width) + ' ' + str(int(width/2)) + ' ' + str(height))
    out = cv2.VideoWriter('output/' + file_name + '.avi', fourcc, video.get(5), (224, 224))
# out = cv2.VideoWriter('output.avi', fourcc, video.get(5), (int(width/2), height))

    i = 0
    while(video.isOpened()):
        flag, frame = video.read()

        if flag:
            # print('True ' + str(i))
            print(file_name, 'frame:', i)
            i = i + 1
            n_frame = urfd_crop_depth_info(frame, width, height)
            n_frame = resize_frame(n_frame, 299, 299)
            # print('s_Shape: ' + str(n_frame.shape))

            # aux = cv2.cvtColor(n_frame, cv2.COLOR_BGR2GRAY)
            # print('g_shape: ' + str(aux.shape) + ' ' + str(aux.dtype))

            # print('Extracting ' + str(i))
            # if i == 0:
            start_time = time.time()
            n_frame = extract_saliency(n_frame, i)
            end_time = time.time()

            print('Time: ' + str(end_time - start_time))

            n_frame = resize_frame(n_frame, 224, 224)
            out.write(n_frame)

            # extract saliency
            # write frame to output file
        else:
            break

    video.release()
    out.release()

def iterate_over_folder(path):

    files = os.listdir(path)

    for file in files:

        print(str(file))
        extract_from_video(file)

iterate_over_folder('input')

# Load the image
# ori = cv2.imread('/home/leite/Dropbox/entry.png')
# im = LoadImage(ori)
# # im = LoadImage('./fall-01-cam0-rgb-150.png')

# # Show the image
# # ShowImage(im)

# # Make a prediction.
# prediction_class = sess.run(prediction, feed_dict = {images: [im]})[0]

# print("Prediction class: " + str(prediction_class))  # Should be a doberman, class idx = 237

# # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
# integrated_gradients = saliency.IntegratedGradients(graph, sess, y, images)

# # Baseline is a black image.
# baseline = np.zeros(im.shape)
# baseline.fill(-1)

# # Compute the vanilla mask and the smoothed mask.
# # vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
# #   im, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)
# # Smoothed mask for integrated gradients will take a while since we are doing nsamples * nsamples computations.
# smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
  # im, feed_dict = {neuron_selector: prediction_class}, x_steps=25, x_baseline=baseline)

# # Call the visualization methods to convert the 3D tensors to 2D grayscale.
# # vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
# smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)

# # Set up matplot lib figures.
# ROWS = 1
# COLS = 2
# UPSCALE_FACTOR = 10
# P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# # Render the saliency masks.
# # ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Integrated Gradients', ax=P.subplot(ROWS, COLS, 1))
# ShowGrayscaleImage(smoothgrad_mask_grayscale, title='Smoothgrad Integrated Gradients', ax=P.subplot(ROWS, COLS, 2))

# # vanilla_mask_grayscale *= (255 / vanilla_mask_grayscale.max())
# smoothgrad_mask_grayscale *= (255 / smoothgrad_mask_grayscale.max())

# # temp_con = np.concatenate((vanilla_mask_grayscale, smoothgrad_mask_grayscale), axis=1)
# # temp_con = np.concatenate((temp_im, temp_con), axis=1)
# # result_con = np.concatenate((result_con, temp_con), axis=0)

# cv2.imwrite('Output.png', smoothgrad_mask_grayscale)
