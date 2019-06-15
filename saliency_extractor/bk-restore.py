import tensorflow as tf
import numpy as np
import sys
import cv2
import saliency

from nets import inception_v3

META_CKPT = '/home/leite/workspace/weights/inception_v3_imagenet_urfd.ckpt.meta'
CKPT_FILE = '/home/leite/workspace/weights/inception_v3_imagenet_urfd.ckpt'

def LoadImage():
    path = str(sys.argv[1])

    frame = cv2.imread(path)
    w, h, c = frame.shape
    frame = cv2.resize(frame, (299, 299), interpolation=cv2.INTER_LINEAR)
    im = np.asarray(frame)
    
    return im / 127.5 - 1.0

saver = tf.train.import_meta_graph(META_CKPT)
saver.restore(sess, CKPT_FILE)

graph = tf.get_default_graph()

images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')

with tf.Session() as sess:

    # saver = tf.train.import_meta_graph(META_CKPT)
    # print(CKPT_FILE)
    # saver.restore(sess, CKPT_FILE)

    # graph = tf.get_default_graph()
    
    # images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    _, end_points = inception_v3.inception_v3(images,
                                              is_training = False,
                                              num_classes = 2)

    # logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')

    neuron_selector = tf.placeholder(tf.int32)
    y = logits[0][neuron_selector]

    prediction = tf.argmax(logits, 1)

    init_op = tf.global_variables_initializer()
    integrated_gradients = saliency.IntegratedGradients(graph, sess, y, images)

    image = LoadImage()

    print(image.shape)

    prediction_class = sess.run(prediction, feed_dict={images:[image]})[0]

    print(prediction_class)
