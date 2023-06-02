import sys
sys.path.insert(0, './facenet/src')

import facenet

import tensorflow as tf
import numpy as np
import cv2
import os
from scipy import misc
from skimage.transform import resize
from align import detect_face
import imageio

import warnings #bruh there are 10 million deprecation warnings
warnings.filterwarnings("ignore")

def load_and_align_image(file_path, image_size=160, margin=32, gpu_memory_fraction=1.0):
    # Create a list of input images
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Read image
    img = imageio.imread(os.path.expanduser(file_path))

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
  
    # Detect face in the image
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    det = np.squeeze(bounding_boxes[0,0:4])

    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
    bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]

    # Resize image to the model input size
    aligned = resize(img, (image_size, image_size), mode='reflect')
    aligned = (aligned * 255).astype(np.uint8)

    prewhitened = facenet.prewhiten(aligned)
    return prewhitened

def calc_emb(model_path, aligned_images):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model_path)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            feed_dict = { images_placeholder: aligned_images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb

# Path to the image
image_path = 'facenet/ex/test_image.jpg'
model_path = 'facenet_model_20180402-114759'

aligned_image = load_and_align_image(image_path)
embedded_image = calc_emb(model_path, [aligned_image])

print('Embedded image shape: ', embedded_image.shape)
print('Embedded image: ', embedded_image)


