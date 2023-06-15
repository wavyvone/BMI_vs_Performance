import h5py
import numpy as np
import sys
sys.path.insert(0, './src')
import facenet
import tensorflow as tf
import os
from scipy import misc
from skimage.transform import resize
from align import detect_face
import pandas as pd
import imageio
import cv2

def modify_and_save_csv(csv_file_path, folder_path):
    '''
    opens morph.csv file, creates bmi column
    returns a list in this format
    [(file_path1, bmi1), (file_path2, bmi2) ...]
    '''
    
    df = pd.read_csv(csv_file_path)
    #converts height into height into feet and inches
    df[['feet', 'inches']] = df['height'].str.split("'", expand=True)
    df["inches"] = df["inches"].str[:-1]

    #removes empty rows
    df = df[df['feet'] != '']

    #calculates height in inches
    df['height_inches'] = df['feet'].astype(int)*12 + df['inches'].astype(int)

    #adds bmi column
    df['bmi'] = 703 * df['weight'].astype(int) / df['height_inches'].astype(int)**2

    # Convert bookid to str to match with image names and add .jpg extension
    df['bookid'] = df['bookid'].astype(str) + '.jpg'
    
    # Add absolute path to bookid column
    df['bookid'] = df['bookid'].apply(lambda x: os.path.join(folder_path, x))
    
    # Filter out entries where file does not exist
    df = df[df['bookid'].apply(os.path.exists)]

    # Save modified dataframe to new csv file
    new_file_path = csv_file_path[:-4] + '_modified.csv'
    df.to_csv(new_file_path, index=False)




def load_and_align_images(filepaths, image_size=160, margin=32, gpu_memory_fraction=1.0):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    aligned_images = []

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
            
            for i, filepath in enumerate(filepaths):
                try:
                    img = imageio.imread(os.path.expanduser(filepath))

                    # If image has 4 channels (RGBA), convert it to 3 channels (RGB)
                    if img.shape[-1] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

                    # Detect face in the image
                    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    if len(bounding_boxes) < 1:
                        print("No face detected in image %s" % filepath)
                        continue
                    #if len(bounding_boxes) > 1:
                        #print("Too many faces detected in image %s" % filepath)
                        #continue
                    det = np.squeeze(bounding_boxes[0,0:4])

                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-margin/2, 0)
                    bb[1] = np.maximum(det[1]-margin/2, 0)
                    bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
                    bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    
                    # Resize image to the model input size
                    aligned = resize(cropped, (image_size, image_size), mode='reflect')
                    aligned = (aligned * 255).astype(np.uint8)

                    prewhitened = facenet.prewhiten(aligned)
                    aligned_images.append(prewhitened)
                except Exception as e:
                    print("An error occurred while processing image " +str(filepath))

    return np.array(aligned_images)

class DataGenerator:
    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
    
    def generate(self):
        num_batches = len(self.images) // self.batch_size
        for i in range(num_batches):
            batch_images = self.images[i*self.batch_size:(i+1)*self.batch_size]
            batch_labels = self.labels[i*self.batch_size:(i+1)*self.batch_size]
            yield batch_images, batch_labels



class BMI_Estimator:
    def __init__(self, feature_size, num_hidden_units_1, num_hidden_units_2, num_hidden_units_3, dropout_rate, learning_rate=0.001):
        self.feature_size = feature_size
        self.num_hidden_units_1 = num_hidden_units_1
        self.num_hidden_units_2 = num_hidden_units_2
        self.num_hidden_units_3 = num_hidden_units_3
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self._build_model()
        self.saver = tf.train.Saver()    

    def _build_model(self):
        self.features = tf.placeholder(tf.float32, shape=[None, self.feature_size])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])

        fc1 = tf.layers.dense(self.features, self.num_hidden_units_1, activation=tf.nn.leaky_relu)
        fc2 = tf.layers.dense(fc1, self.num_hidden_units_2, activation=tf.nn.leaky_relu)
        dropout = tf.layers.dropout(fc2, rate=self.dropout_rate)
        fc3 = tf.layers.dense(dropout, self.num_hidden_units_3, activation=tf.nn.leaky_relu)
        self.predictions = tf.layers.dense(fc3, 1)
        
        self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    
    def train(self, sess, features, labels):
        feed_dict = {self.features: features, self.labels: labels}
        _, loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        return loss
    
    def predict(self, sess, features):
        feed_dict = {self.features: features}
        return sess.run(self.predictions, feed_dict=feed_dict)

        
    def save_model(self, sess, epoch, base_path="model_checkpoint"):
        # Define the save path
        save_path = "{}/model_epoch_{}/".format(base_path, epoch)

        # Check if the save directory exists, if not, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save the model in the directory specific to the epoch
        self.saver.save(sess, save_path + "model.ckpt")


def store_processed_image(filepaths):
    print('Saving Dataset')
    all_images = load_and_align_images(filepaths)

    with h5py.File('load_imgs.h5', 'w') as f:
        f.create_dataset('data', data=all_images)



def pre_process_images(filepaths, labels):
    print('processing Dataset')
    #all_images = load_and_align_images(filepaths)
    with h5py.File('load_imgs.h5', 'r') as f:
        all_images = f['data'][:]

    all_labels = np.array(labels).reshape(-1, 1)

    indices = np.arange(len(all_images))
    np.random.shuffle(indices)
    
    # Split into training and validation
    split_idx = int(0.8 * len(all_images))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_images = all_images[train_indices]
    train_labels = all_labels[train_indices]

    val_images = all_images[val_indices]
    val_labels = all_labels[val_indices]
    
    return train_images, train_labels, val_images, val_labels


# Load filepaths and labels
folder_path = 'data/morph'
csv_file_path = 'data/kpop_morphe_modified.csv'

#you only have to run this once!
#modify_and_save_csv('data/full.csv', folder_path)

df = pd.read_csv(csv_file_path)
filepaths = df['bookid'].tolist()
labels = df['bmi'].tolist()
labels = np.array(labels).reshape(-1, 1)

#run this once
#store_processed_image(filepaths)

# Split data into training and validation sets
train_images, train_labels, val_images, val_labels = pre_process_images(filepaths, labels)

# Create data generators
batch_size = 256  # adjust as needed
train_data_generator = DataGenerator(train_images, train_labels, batch_size)
val_data_generator = DataGenerator(val_images, val_labels, batch_size)

# Load the model
with tf.Graph().as_default():
    with tf.Session() as sess:
        model_path = 'facenet_model_20180402-11475'
        facenet.load_model(model_path)
        
        model_number = 1
        bmi_model_path = 'model_checkpoint' + str(model_number)
        while os.path.exists(bmi_model_path):
            model_number += 1
            bmi_model_path = 'model_checkpoint' + str(model_number)

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        print('calculating facial feature vectors')
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # Initialize BMI Estimator model
        feature_size = 512
        num_hidden_units_1 = 512
        num_hidden_units_2 = 256
        num_hidden_units_3 = 128
        dropout_rate = 0.5
        learning_rate = 1e-4  # Adjust as needed
        model = BMI_Estimator(feature_size, num_hidden_units_1, num_hidden_units_2, num_hidden_units_3, dropout_rate, learning_rate)
        sess.run(tf.global_variables_initializer())

        num_epochs = 25

        # Train the model
        print('begin training')
        for epoch in range(num_epochs):
            training_losses = []
            for images, labels in train_data_generator.generate():
                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                features = sess.run(embeddings, feed_dict=feed_dict)

                # Train the model
                loss = model.train(sess, features, labels)
                training_losses.append(loss)

            print("Epoch: {}, Training Loss: {}".format(epoch, np.mean(training_losses)))

            # Validation
            val_losses = []
            for images, labels in val_data_generator.generate():
                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                features = sess.run(embeddings, feed_dict=feed_dict)

                # Compute validation loss
                predictions = model.predict(sess, features)
                val_loss = sess.run(model.loss, feed_dict={model.features: features, model.labels: labels})
                val_losses.append(val_loss)

            print("Epoch: {}, Validation Loss: {}".format(epoch, np.mean(val_losses)))
            model.save_model(sess, epoch, bmi_model_path)
