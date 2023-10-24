import streamlit as st

import os
import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
#from google.colab import drive

#import cv2 //AK commented out.
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

#Custom import
from pages.scripts.content.util import get_base_path, get_content_folder, get_content_path
from pages.scripts.content.util import get_content_pkl_path, get_classified_lable_file_path, is_debug

#---------------------------------------------------------------------------

#drive.mount('/content/drive', force_remount=True) DB
# Load the value of 'classified_label' from Google Drive
#with open('/content/drive/MyDrive/Phase-6_ML_Classifier_app_py/classified_label.txt', 'r') as f:
#    classified_label = f.read()

#New............................AK    
with open(get_classified_lable_file_path(), 'r') as f:
    classified_label = f.read()

if is_debug() == True:
  # Now you can use 'classified_label' in app2.py
  print("The classified label is:", classified_label)
  #st.write("The classified label:", classified_label)

action_folders = ['C-Archery', 'C-Basketball', 'C-Biking', 'C-CricketShot', 'C-HorseRace', 'C-IceDancing', 'C-Kayaking', 'C-LongJump', 'C-MilitaryParade', 'C-PlayingTabla']

for action_folder in action_folders:
  action_class = action_folder.split('-')[1]
  if action_class == classified_label:
    L = action_class
    break
#---------------------------------------------------------------------------
#AK commented out this
# user_input1 = st.text_input("Enter the batch size","Your desired batch size ")
# batch_size = int(user_input1)
# user_input2 = st.text_input("Enter the no. of epochs","Your desired no. of epochs ")
# num_epochs = int(user_input2)
#---------------------------------------------------------------------------

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize_with_crop_or_pad(img, IMG_H, IMG_W)
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5
    return img
#---------------------------------------------------------------------------

def tf_dataset(images_path, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(buffer_size=10240)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
#---------------------------------------------------------------------------

class GAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for _ in range(2):
            ## Train the discriminator
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)
            generated_labels = tf.zeros((batch_size, 1))

            with tf.GradientTape() as ftape:
                predictions = self.discriminator(generated_images)
                d1_loss = self.loss_fn(generated_labels, predictions)
            grads = ftape.gradient(d1_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            ## Train the discriminator
            labels = tf.ones((batch_size, 1))

            with tf.GradientTape() as rtape:
                predictions = self.discriminator(real_images)
                d2_loss = self.loss_fn(labels, predictions)
            grads = rtape.gradient(d2_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        ## Train the generator
        tf.random.set_seed(5)
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as gtape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = gtape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d1_loss": d1_loss, "d2_loss": d2_loss, "g_loss": g_loss}
#---------------------------------------------------------------------------

def save_plot(examples, epoch, n):
    gen_images = []
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis("off")
        pyplot.imshow(examples[i])  ## pyplot.imshow(np.squeeze(examples[i], axis=-1))
        gen_images.append(examples[i])
    filename = f"/content/generated_plot_epoch-{epoch+1}.png"
    pyplot.savefig(filename)
    pyplot.close()
    return gen_images
#---------------------------------------------------------------------------

def gen_video(frame_directory, num_epochs, output_video_path, gen_images):
    frames = []

    # Load 25 frames from the last image (based on num_epochs)
    for i in range(25):
        image_path = f'{frame_directory}/generated_plot_epoch-{num_epochs}.png'
        #img = cv2.imread(image_path)
        img = cv2.imread(gen_images[i])
        frames.append(img)

    # Get the height and width of the frames
    height, width, layers = frames[0].shape

    # Define the video codec and create a VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # We can change the codec as needed
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (width, height))

    # Write frames to the video
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter
    out.release()

    print(f"Video saved at: {output_video_path}")
#---------------------------------------------------------------------------

def play_video(video_path, fps=25, width=120, height=100, loop=True):
    video_clip = VideoFileClip(video_path)

    # If loop is set to True, the video will loop indefinitely
    if loop:
        video_clip = video_clip.loop(duration=video_clip.duration)

    # Displaying the video clip with custom parameters
    video_clip.ipython_display(fps=fps, width=width, height=height, autoplay=True, loop=True, ctrls=True)
#---------------------------------------------------------------------------

if __name__ == "__main__":
    if st.button('Generate Video'):
    ## Hyperparameters
    #batch_size = 32
      latent_dim = 128
    #num_epochs = 6
    #images_path = glob("data/*")
    images_path = glob('/content/drive/MyDrive/Phase-6_ML_Classifier_app_py/C-'+L+'/frames/*')

    #d_model = build_discriminator()
    #g_model = build_generator(latent_dim)
    if L == 'Archery':
      Archery_d_model.load_weights("/content/saved_model/Archery_d_model.h5")
      Archery_g_model.load_weights("/content/saved_model/Archery_g_model.h5")
      gan = GAN(Archery_d_model, Archery_g_model, latent_dim)
    elif L == 'Basketball':
      Basketball_d_model.load_weights("/content/saved_model/Basketball_d_model.h5")
      Basketball_g_model.load_weights("/content/saved_model/Basketball_g_model.h5")
      gan = GAN(Basketball_d_model, Basketball_g_model, latent_dim)
    elif L == 'Biking':
      Biking_d_model.load_weights("/content/saved_model/Biking_d_model.h5")
      Biking_g_model.load_weights("/content/saved_model/Biking_g_model.h5")
    elif L == 'CricketShot':
      CricketShot_d_model.load_weights("/content/saved_model/CricketShot_d_model.h5")
      CricketShot_g_model.load_weights("/content/saved_model/CricketShot_g_model.h5")
      gan = GAN(CricketShot_d_model, CricketShot_g_model, latent_dim)
    elif L == 'HorseRace':
      HorseRace_d_model.load_weights("/content/saved_model/HorseRace_d_model.h5")
      HorseRace_g_model.load_weights("/content/saved_model/HorseRace_g_model.h5")
      gan = GAN(HorseRace_d_model, HorseRace_g_model, latent_dim)
    elif L == 'IceDancing':
      IceDancing_d_model.load_weights("/content/saved_model/IceDancing_d_model.h5")
      IceDancing_g_model.load_weights("/content/saved_model/IceDancing_g_model.h5")
    elif L == 'Kayaking':
      Kayaking_d_model.load_weights("/content/saved_model/Kayaking_d_model.h5")
      Kayaking_g_model.load_weights("/content/saved_model/Kayaking_g_model.h5")
      gan = GAN(Kayaking_d_model, Kayaking_g_model, latent_dim)
    elif L == 'LongJump':
      LongJump_d_model.load_weights("/content/saved_model/LongJump_d_model.h5")
      LongJump_g_model.load_weights("/content/saved_model/LongJump_g_model.h5")
      gan = GAN(LongJump_d_model, LongJump_g_model, latent_dim)
    elif L == 'MilitaryParade':
      MilitaryParade_d_model.load_weights("/content/saved_model/MilitaryParade_d_model.h5")
      MilitaryParade_g_model.load_weights("/content/saved_model/MilitaryParade_g_model.h5")
      gan = GAN(MilitaryParade_d_model, MilitaryParade_g_model, latent_dim)
    elif L == 'PlayingTabla':
      PlayingTabla_d_model.load_weights("/content/saved_model/PlayingTabla_d_model.h5")
      PlayingTabla_g_model.load_weights("/content/saved_model/PlayingTabla_g_model.h5")
      gan = GAN(PlayingTabla_d_model, PlayingTabla_g_model, latent_dim)
    else:
      print('Invalid action label')

    #d_model.summary()
    #g_model.summary()

    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(d_optimizer, g_optimizer, bce_loss_fn)

    images_dataset = tf_dataset(images_path, batch_size)

    for epoch in range(num_epochs):
        gan.fit(images_dataset, epochs=1)
        #if epoch == num_epochs-1:
          #g_model.save("/content/g_model.h5")
          #d_model.save("/content/d_model.h5")

        n_samples = 25
        noise = np.random.normal(size=(n_samples, latent_dim))
        if L == 'Archery':
          examples = Archery_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'Basketball':
          examples = Basketball_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'Biking':
          examples = Biking_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'CricketShot':
          examples = CricketShot_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'HorseRace':
          examples = HorseRace_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'IceDancing':
          examples = IceDancing_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'LongJump':
          examples = LongJump_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'MilitaryParade':
          examples = MilitaryParade_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'PlayingTabla':
          examples = PlayingTabla_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        else:
          print('Invalid frames')

    frame_directory = '/content/'
    output_video_path = '/content/generated_video.avi'
    gen_video(frame_directory, num_epochs, output_video_path)
    play_video(output_video_path)

def generate_video(my_batch_size, my_ephoc_size) :
    ## Hyperparameters
    #batch_size = 32
    batch_size = my_batch_size
    latent_dim = 128
    #num_epochs = 6
    num_epochs = my_ephoc_size 
    
    #images_path = glob("data/*")
    images_path = glob('/content/drive/MyDrive/Phase-6_ML_Classifier_app_py/C-'+L+'/frames/*')

    #d_model = build_discriminator()
    #g_model = build_generator(latent_dim)
    if L == 'Archery':
      Archery_d_model.load_weights("/content/saved_model/Archery_d_model.h5")
      Archery_g_model.load_weights("/content/saved_model/Archery_g_model.h5")
      gan = GAN(Archery_d_model, Archery_g_model, latent_dim)
    elif L == 'Basketball':
      Basketball_d_model.load_weights("/content/saved_model/Basketball_d_model.h5")
      Basketball_g_model.load_weights("/content/saved_model/Basketball_g_model.h5")
      gan = GAN(Basketball_d_model, Basketball_g_model, latent_dim)
    elif L == 'Biking':
      Biking_d_model.load_weights("/content/saved_model/Biking_d_model.h5")
      Biking_g_model.load_weights("/content/saved_model/Biking_g_model.h5")
    elif L == 'CricketShot':
      CricketShot_d_model.load_weights("/content/saved_model/CricketShot_d_model.h5")
      CricketShot_g_model.load_weights("/content/saved_model/CricketShot_g_model.h5")
      gan = GAN(CricketShot_d_model, CricketShot_g_model, latent_dim)
    elif L == 'HorseRace':
      HorseRace_d_model.load_weights("/content/saved_model/HorseRace_d_model.h5")
      HorseRace_g_model.load_weights("/content/saved_model/HorseRace_g_model.h5")
      gan = GAN(HorseRace_d_model, HorseRace_g_model, latent_dim)
    elif L == 'IceDancing':
      IceDancing_d_model.load_weights("/content/saved_model/IceDancing_d_model.h5")
      IceDancing_g_model.load_weights("/content/saved_model/IceDancing_g_model.h5")
    elif L == 'Kayaking':
      Kayaking_d_model.load_weights("/content/saved_model/Kayaking_d_model.h5")
      Kayaking_g_model.load_weights("/content/saved_model/Kayaking_g_model.h5")
      gan = GAN(Kayaking_d_model, Kayaking_g_model, latent_dim)
    elif L == 'LongJump':
      LongJump_d_model.load_weights("/content/saved_model/LongJump_d_model.h5")
      LongJump_g_model.load_weights("/content/saved_model/LongJump_g_model.h5")
      gan = GAN(LongJump_d_model, LongJump_g_model, latent_dim)
    elif L == 'MilitaryParade':
      MilitaryParade_d_model.load_weights("/content/saved_model/MilitaryParade_d_model.h5")
      MilitaryParade_g_model.load_weights("/content/saved_model/MilitaryParade_g_model.h5")
      gan = GAN(MilitaryParade_d_model, MilitaryParade_g_model, latent_dim)
    elif L == 'PlayingTabla':
      PlayingTabla_d_model.load_weights("/content/saved_model/PlayingTabla_d_model.h5")
      PlayingTabla_g_model.load_weights("/content/saved_model/PlayingTabla_g_model.h5")
      gan = GAN(PlayingTabla_d_model, PlayingTabla_g_model, latent_dim)
    else:
      print('Invalid action label')

    #d_model.summary()
    #g_model.summary()

    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(d_optimizer, g_optimizer, bce_loss_fn)

    images_dataset = tf_dataset(images_path, batch_size)

    for epoch in range(num_epochs):
        gan.fit(images_dataset, epochs=1)
        #if epoch == num_epochs-1:
          #g_model.save("/content/g_model.h5")
          #d_model.save("/content/d_model.h5")

        n_samples = 25
        noise = np.random.normal(size=(n_samples, latent_dim))
        if L == 'Archery':
          examples = Archery_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'Basketball':
          examples = Basketball_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'Biking':
          examples = Biking_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'CricketShot':
          examples = CricketShot_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'HorseRace':
          examples = HorseRace_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'IceDancing':
          examples = IceDancing_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'LongJump':
          examples = LongJump_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'MilitaryParade':
          examples = MilitaryParade_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        elif L == 'PlayingTabla':
          examples = PlayingTabla_g_model.predict(noise)
          save_plot(examples, epoch, int(np.sqrt(n_samples)))
        else:
          print('Invalid frames')

    frame_directory = '/content/'
    output_video_path = '/content/generated_video.avi'
    gen_video(frame_directory, num_epochs, output_video_path)
    play_video(output_video_path)