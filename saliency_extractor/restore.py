import tensorflow as tf
import numpy as np
import cv2 as cv
import tensorflow.contrib.slim.nets

INCEPTION_WEIGHTS = '/home/leite/workspace/weights/inception_v3_imagenet_urfd.ckpt'
NUM_CLASSES = 2
# INCEPTION_WEIGHTS = './inception_v3.ckpt'
# NUM_CLASSES = 1001

TB_DIR = '/home/leite/workspace/tb_logdir/'

# Construction phase -----------------------------------------------------------

input_dims = (None, 299, 299, 3)
tf_X = tf.placeholder(tf.float32, shape=input_dims, name='X')
tf_Y = tf.placeholder(tf.int32, shape=[None], name='Y')
tf_is_training = tf.placeholder_with_default(False, shape=None, name='is_training')

scaled_inputs = tf.div(tf_X, 255., name='rescaled_inputs')

ARG_SCOPE = tf.contrib.slim.nets.inception.inception_v3_arg_scope()
with tf.contrib.framework.arg_scope(ARG_SCOPE):
    tf_logits, end_points = tf.contrib.slim.nets.inception.inception_v3(
            scaled_inputs,
            num_classes=NUM_CLASSES,
            is_training=tf_is_training,
            dropout_keep_prob=0.8)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

file_writer = tf.summary.FileWriter(TB_DIR, tf.get_default_graph())

# Execution phase --------------------------------------------------------------

def load_image():
    # image = cv.imread('/home/leite/Pictures/tinca.jpg')
    # image = cv.imread('/home/leite/Pictures/adl-01-cam0/frame_00001.jpg')
    image = []
    # for i in range(100, 151):
    for i in range(10, 31):
        # frame = cv.imread('/home/leite/Pictures/adl-01-cam0/frame_00' + str(i) + '.jpg')
        frame = cv.imread('/home/leite/Pictures/fall-01-cam0/frame_000' + str(i) + '.jpg')
        frame = cv.resize(frame, (299, 299))
        # print(np.max(frame))
        image.append(frame)
    return image

with tf.Session() as sess:
    
    saver.restore(sess, INCEPTION_WEIGHTS)
    input = load_image()

    z = tf_logits.eval(feed_dict={tf_X: input})
    for i in range(len(z)):
        print('ImageNet ID + 1:', np.argmax(z[i]), z[i][np.argmax(z[i])], z[i])
