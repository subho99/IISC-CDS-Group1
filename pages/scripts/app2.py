import sys
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
import moviepy.editor as moviepy


import cv2
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

#Custom import
from pages.scripts.content.util import get_content_path, is_debug, is_info

#---------------------------------------------------------------------------
IMG_H = 64
IMG_W = 64
IMG_C = 3  ## Change this to 1 for grayscale.
w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

#drive.mount('/content/drive', force_remount=True)
# Load the value of 'classified_label' from Google Drive
def load_classified_label(classified_label):
  selected_action = ""
  if is_debug() == True:
    print("The classified label is:", classified_label)

  action_folders = ['C-Archery', 'C-Basketball', 'C-Biking', 'C-CricketShot', 'C-HorseRace', 'C-IceDancing', 'C-Kayaking', 'C-LongJump', 'C-MilitaryParade', 'C-PlayingTabla']

  for action_folder in action_folders:
    action_class = action_folder.split('-')[1]
    if action_class == classified_label:
      L = action_class
      selected_action = action_class
      if is_debug() == True:
          print("Debug Selected classified label L  n-->", L)
      break
  return selected_action
#---------------------------------------------------------------------------

# user_input1 = st.text_input("Enter the batch size","Your desired batch size ") # AK commented out
# batch_size = int(user_input1)
# user_input2 = st.text_input("Enter the no. of epochs","Your desired no. of epochs ")
# num_epochs = int(user_input2)
#---------------------------------------------------------------------------

def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2, activation=True):
    x = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding=padding,
        strides=strides,
    )(inputs)

    if activation:
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
    return x
#---------------------------------------------------------------------------

def load_image(image_path):
    if is_debug()== True:
       print("  image_path --->", image_path); 
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
def save_plot_new(examples):
    gen_images = []
    examples = (examples + 1) / 2.0
    for i in range(len(examples)):
        # cv2.imwrite(f"{i}.png", examples[i])
        pyplot.axis("off")
        pyplot.imshow(examples[i])  ## pyplot.imshow(np.squeeze(examples[i], axis=-1))
        if os.path.exists(os.path.join(get_content_path(), "generated_plot_epoch-{epoch+1}/")) == False:
          os.makedirs(f"/content/generated_plot_epoch-{epoch+1}/")

        c = cv2.imread(f"{i}.png")
        gen_images.append(c)
    return gen_images



def save_plot(examples, epoch, n, isvideo):
    gen_images = []
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        #pyplot.subplot(n, n, i+1)
        pyplot.axis("off")
        pyplot.imshow(examples[i])  ## pyplot.imshow(np.squeeze(examples[i], axis=-1))
        
        #qf = pyplot.imshow(examples[i])
        #gen_images.append(examples[i])
        if os.path.exists(f"/content/generated_plot_epoch-{epoch+1}/") == False:
          os.makedirs(f"/content/generated_plot_epoch-{epoch+1}/")
        filename = f"/content/generated_plot_epoch-{epoch+1}/'{i}.png"
        pyplot.savefig(filename)
        pyplot.close()
        if isvideo:
          cv2.imwrite(f"{i}.png", examples[i])
          c = cv2.imread(f"{i}.png")
          gen_images.append(c)
          #c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

    return gen_images
#---------------------------------------------------------------------------

def gen_video(output_video_path):
    #frames =  [cv2.imread(os.path.join(get_content_path(), "generated_images/") + i) for i in os.listdir("C:/Users/Admin/iisc-capstone/iisc-group1-capstonetemp/pages/scripts/content/generated_images")]
    frames =  [cv2.imread(os.path.join(get_content_path(), "generated_images/") + i) for i in os.listdir(os.path.join(get_content_path(), "generated_images"))]

    # cv2.imshow(frames[0])
    
    # Get the height and width of the frames
    height, width, layers = frames[0].shape

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (width, height))

    # Write frames to the video
    for frame in frames:
       # print("Frame from gen_video()", frame)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #st.image(rgb_image, use_column_width=True, channels="BGR")
        out.write(rgb_image)

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

# if __name__ == "__main__":
#     if st.button('Generate Video'):
#     ## Hyperparameters
#     #batch_size = 32
#     latent_dim = 128
#     #num_epochs = 6
#     #images_path = glob("data/*")
#     images_path = glob('/content/drive/MyDrive/Phase-6_ML_Classifier_app_py/C-'+L+'/frames/*')
#     isvideo = True

#     #d_model = build_discriminator()
#     #g_model = build_generator(latent_dim)
#     if L == 'Archery':
#       Archery_d_model.load_weights("/content/saved_model/Archery_d_model.h5")
#       Archery_g_model.load_weights("/content/saved_model/Archery_g_model.h5")
#       gan = GAN(Archery_d_model, Archery_g_model, latent_dim)
#     elif L == 'Basketball':
#       Basketball_d_model.load_weights("/content/saved_model/Basketball_d_model.h5")
#       Basketball_g_model.load_weights("/content/saved_model/Basketball_g_model.h5")
#       gan = GAN(Basketball_d_model, Basketball_g_model, latent_dim)
#     elif L == 'Biking':
#       Biking_d_model.load_weights("/content/saved_model/Biking_d_model.h5")
#       Biking_g_model.load_weights("/content/saved_model/Biking_g_model.h5")
#     elif L == 'CricketShot':
#       CricketShot_d_model.load_weights("/content/saved_model/CricketShot_d_model.h5")
#       CricketShot_g_model.load_weights("/content/saved_model/CricketShot_g_model.h5")
#       gan = GAN(CricketShot_d_model, CricketShot_g_model, latent_dim)
#     elif L == 'HorseRace':
#       HorseRace_d_model.load_weights("/content/saved_model/HorseRace_d_model.h5")
#       HorseRace_g_model.load_weights("/content/saved_model/HorseRace_g_model.h5")
#       gan = GAN(HorseRace_d_model, HorseRace_g_model, latent_dim)
#     elif L == 'IceDancing':
#       IceDancing_d_model.load_weights("/content/saved_model/IceDancing_d_model.h5")
#       IceDancing_g_model.load_weights("/content/saved_model/IceDancing_g_model.h5")
#     elif L == 'Kayaking':
#       Kayaking_d_model.load_weights("/content/saved_model/Kayaking_d_model.h5")
#       Kayaking_g_model.load_weights("/content/saved_model/Kayaking_g_model.h5")
#       gan = GAN(Kayaking_d_model, Kayaking_g_model, latent_dim)
#     elif L == 'LongJump':
#       LongJump_d_model.load_weights("/content/saved_model/LongJump_d_model.h5")
#       LongJump_g_model.load_weights("/content/saved_model/LongJump_g_model.h5")
#       gan = GAN(LongJump_d_model, LongJump_g_model, latent_dim)
#     elif L == 'MilitaryParade':
#       MilitaryParade_d_model.load_weights("/content/saved_model/MilitaryParade_d_model.h5")
#       MilitaryParade_g_model.load_weights("/content/saved_model/MilitaryParade_g_model.h5")
#       gan = GAN(MilitaryParade_d_model, MilitaryParade_g_model, latent_dim)
#     elif L == 'PlayingTabla':
#       PlayingTabla_d_model.load_weights("/content/saved_model/PlayingTabla_d_model.h5")
#       PlayingTabla_g_model.load_weights("/content/saved_model/PlayingTabla_g_model.h5")
#       gan = GAN(PlayingTabla_d_model, PlayingTabla_g_model, latent_dim)
#     else:
#       print('Invalid action label')

#     #d_model.summary()
#     #g_model.summary()

#     bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
#     d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
#     g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
#     gan.compile(d_optimizer, g_optimizer, bce_loss_fn)

#     images_dataset = tf_dataset(images_path, batch_size)

#     for epoch in range(num_epochs):
#         gan.fit(images_dataset, epochs=1)
#         #if epoch == num_epochs-1:
#           #g_model.save("/content/g_model.h5")
#           #d_model.save("/content/d_model.h5")

#         n_samples = 25
#         noise = np.random.normal(size=(n_samples, latent_dim))
#         if L == 'Archery':
#           examples = Archery_g_model.predict(noise)
#           gen_images = save_plot(examples, epoch, int(np.sqrt(n_samples)), isvideo)
#         elif L == 'Basketball':
#           examples = Basketball_g_model.predict(noise)
#           gen_images = save_plot(examples, epoch, int(np.sqrt(n_samples)), isvideo)
#         elif L == 'Biking':
#           examples = Biking_g_model.predict(noise)
#           gen_images = save_plot(examples, epoch, int(np.sqrt(n_samples)), isvideo)
#         elif L == 'CricketShot':
#           examples = CricketShot_g_model.predict(noise)
#           gen_images = save_plot(examples, epoch, int(np.sqrt(n_samples)), isvideo)
#         elif L == 'HorseRace':
#           examples = HorseRace_g_model.predict(noise)
#           gen_images = save_plot(examples, epoch, int(np.sqrt(n_samples)), isvideo)
#         elif L == 'IceDancing':
#           examples = IceDancing_g_model.predict(noise)
#           gen_images = save_plot(examples, epoch, int(np.sqrt(n_samples)), isvideo)
#         elif L == 'LongJump':
#           examples = LongJump_g_model.predict(noise)
#           gen_images = save_plot(examples, epoch, int(np.sqrt(n_samples)), isvideo)
#         elif L == 'MilitaryParade':
#           examples = MilitaryParade_g_model.predict(noise)
#           gen_images = save_plot(examples, epoch, int(np.sqrt(n_samples)), isvideo)
#         elif L == 'PlayingTabla':
#           examples = PlayingTabla_g_model.predict(noise)
#           gen_images = save_plot(examples, epoch, int(np.sqrt(n_samples)), isvideo)
#         else:
#           print('Invalid frames')

#     frame_directory = '/content/'
#     output_video_path = '/content/generated_video.avi'
#     gen_video(frame_directory, num_epochs, output_video_path, gen_images)
#     play_video(output_video_path)

#---------------------------------------------------------------------------
def deconv_block(inputs, num_filters, kernel_size, strides, bn=True):
    x = Conv2DTranspose(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding="same",
        strides=strides,
        use_bias=False
        )(inputs)

    if bn:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
    return x

#---------------------------------------------------------------------------

def save_examples(examples):

  counter = 0 
  for image in examples:
      pyplot.axis("off")
      pyplot.imshow(image)  ## pyplot.imshow(np.squeeze(examples[i], axis=-1))
      filename = os.path.join(get_content_path(), "generated_images") + "/"+ str(counter) + ".png"
      if is_debug() == True:
        print( "filename from save_examples  --------------->", filename)
      pyplot.savefig(filename)
      counter = counter + 1 
#---------------------------------------------------------------------------

def build_generator(latent_dim):
    f = [2**i for i in range(5)][::-1]
    filters = 32
    output_strides = 16
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides

    noise = Input(shape=(latent_dim,), name="generator_noise_input")

    x = Dense(f[0] * filters * h_output * w_output, use_bias=False)(noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((h_output, w_output, 16 * filters))(x)

    for i in range(1, 5):
        x = deconv_block(x,
            num_filters=f[i] * filters,
            kernel_size=5,
            strides=2,
            bn=True
        )

    x = conv_block(x,
        num_filters=3,  ## Change this to 1 for grayscale.
        kernel_size=5,
        strides=1,
        activation=False
    )
    fake_output = Activation("tanh")(x)

    return Model(noise, fake_output, name="generator")
#---------------------------------------------------------------------------

def build_discriminator():
    f = [2**i for i in range(4)]
    image_input = Input(shape=(IMG_H, IMG_W, IMG_C))
    x = image_input
    filters = 64
    output_strides = 16
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides

    for i in range(0, 4):
        x = conv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2)

    x = Flatten()(x)
    x = Dense(1)(x)

    return Model(image_input, x, name="discriminator")
#---------------------------------------------------------------------------


actions = {
"Archery":  {
    "d_model_file" : "saved_model/Archery_d_model.h5",
    "g_model_file" : "saved_model/Archery_g_model.h5"
  },
# "Basketball": "Mustang",
# "Biking": 1964,
# "CricketShot": 
# "HorseRace":
# "IceDancing" : 
# "Kayaking" :
# "LongJump": 
# "MilitaryParade":
"PlayingTabla": {
    "d_model_file" : "saved_model/PlayingTabla_d_model.h5",
    "g_model_file" : "saved_model/PlayingTabla_g_model.h5"
  }
}

def exit_program():
    print("Exiting the program...")
    sys.exit(0)

def generate_video(classified_label, show_images):
    
    examples =""
    n_samples = 25
    latent_dim = 128
    noise = np.random.normal(size=(n_samples, latent_dim))

    d_model = build_discriminator()
    g_model = build_generator(latent_dim)

    L = load_classified_label(classified_label)
    if L in actions:
      d_model_file = os.path.join(get_content_path(), actions[L]["d_model_file"])
      g_model_file = os.path.join(get_content_path(), actions[L]["g_model_file"])

      if os.path.exists(d_model_file) == True:
        d_model.load_weights(d_model_file)
      else:
        st.info(L + " discriminator model file does not exit")
        exit_program()

      if os.path.exists(g_model_file) == True:
        g_model.load_weights(g_model_file)
        examples = g_model.predict(noise)
        save_examples(examples)

        #Generate the video in avi
        output_video_path = os.path.join(get_content_path(),'generated_videos', 'generated_video.avi')
        if is_debug() == True:
          print("output_video_path.......?" , output_video_path)

        gen_video(output_video_path)

        #Convert avi to mp4
        clip = moviepy.VideoFileClip(output_video_path)
        output_video_path_converted_mp4 = os.path.join(get_content_path(),'generated_videos', 'generated_video.mp4')
        clip.write_videofile(output_video_path_converted_mp4)

        with st.expander("Generated video"): 
          #Show the video
          video_file = open(output_video_path_converted_mp4, 'rb') #enter the filename with filepath
          video_bytes = video_file.read() #reading the file
          st.video(video_bytes, format='video/mp4', start_time=0) #displaying the video

        if show_images:
            with st.expander("Generated images"): 
                for i in os.listdir(os.path.join(get_content_path(), "generated_images")):
                    st.write(os.path.join(get_content_path(), "generated_images/") + i)
                    st.image(cv2.imread(os.path.join(get_content_path(), "generated_images/") + i))   
      else:
         st.info(L + " generator model file does not exit")
         exit_program()
    else:
       st.info(" No trainned model is found for the action " + L)
       exit_program()
      



# def generate_video_old(classified_label):
    
#     L = load_classified_label(classified_label)
  
#     if is_debug() == True:
#        print(" what is set in  L ", L)
#     ## Hyperparameters
#     #batch_size = 32
#     #batch_size = my_batch_size
#     latent_dim = 128
#     #num_epochs = my_ephoc_size
#     #num_epochs = 6
#     #images_path = glob("data/*")
#     if is_info()== True:
#         print("INFO get_content_path() in app2.py --> ", get_content_path())
#     ipath =  os.path.join(get_content_path(),'drive/MyDrive/Phase-6_ML_Classifier_app_py/C-'+L)
#     images_path = glob(ipath +'/frames/*')
#     isvideo = True

#     d_model = build_discriminator()
#     g_model = build_generator(latent_dim)
#     if L == 'Archery':
#       archery_d_model_file = os.path.join(get_content_path(),"saved_model/Archery_d_model.h5")
#       if os.path.exists(archery_d_model_file) == True:
#         d_model.load_weights(archery_d_model_file)
#       else:
#         st.write("archery_d_model_file does not exit")

#       archery_g_model_file = os.path.join(get_content_path(),"saved_model/Archery_g_model.h5")
#       if os.path.exists(archery_g_model_file) == True:
#         g_model.load_weights(archery_g_model_file)
#       else:
#         st.write("archery_g_model_file does not exit")
#     elif L == 'Basketball':
#       d_model.load_weights(os.path.join(get_content_path(),"saved_model/Basketball_d_model.h5"))
#       g_model.load_weights(os.path.join(get_content_path(),"saved_model/Basketball_g_model.h5"))
#     elif L == 'Biking':
#       d_model.load_weights(os.path.join(get_content_path(),"saved_model/Biking_d_model.h5"))
#       g_model.load_weights(os.path.join(get_content_path(),"saved_model/Biking_g_model.h5"))
#     elif L == 'CricketShot':
#       d_model.load_weights(os.path.join(get_content_path(),"saved_model/CricketShot_d_model.h5"))
#       g_model.load_weights(os.path.join(get_content_path(),"saved_model/CricketShot_g_model.h5"))
#     elif L == 'HorseRace':
#       d_model.load_weights(os.path.join(get_content_path(),"saved_model/HorseRace_d_model.h5"))
#       g_model.load_weights(os.path.join(get_content_path(),"saved_model/HorseRace_g_model.h5"))
#     elif L == 'IceDancing':
#       #d_model.load_weights(os.path.join(get_content_path(),"saved_model/IceDancing_d_model.h5"))
#       g_model.load_weights(os.path.join(get_content_path(),"saved_model/IceDancing_g_model.h5"))
#       #g_model.load_weights('https://drive.google.com/uc?export=view&id=1SIg9gRxzu1Lysh3upkQ8-F0hYGaO5CEE')
      
#     elif L == 'Kayaking':
#       d_model.load_weights(os.path.join(get_content_path(),"saved_model/Kayaking_d_model.h5"))
#       g_model.load_weights(os.path.join(get_content_path(),"saved_model/Kayaking_g_model.h5"))
#     elif L == 'LongJump':
#       d_model.load_weights(os.path.join(get_content_path(),"saved_model/LongJump_d_model.h5"))
#       g_model.load_weights(os.path.join(get_content_path(),"saved_model/LongJump_g_model.h5"))
#     elif L == 'MilitaryParade':
#       d_model.load_weights(os.path.join(get_content_path(),"saved_model/MilitaryParade_d_model.h5"))
#       g_model.load_weights(os.path.join(get_content_path(),"saved_model/MilitaryParade_g_model.h5"))
#     elif L == 'PlayingTabla':
#       playingTabla_d_model_file = os.path.join(os.path.join(get_content_path(),"saved_model/PlayingTabla_d_model.h5"))
#       if os.path.exists(playingTabla_d_model_file) == True:
#         d_model.load_weights(playingTabla_d_model_file)
#       else:
#         st.error("playingTabla_d_model_file does not exit")

#       playingTabla_g_model_file = os.path.join(os.path.join(get_content_path(),"saved_model/PlayingTabla_g_model.h5"))
#       if os.path.exists(playingTabla_g_model_file) == True:
#         g_model.load_weights(playingTabla_g_model_file)
#       else:
#         st.error("playingTabla_g_model_file does not exit")
#     else:
#       L = "Invalid"      

#     if L != "Invalid":
#       examples =""
#       n_samples = 25
#       noise = np.random.normal(size=(n_samples, latent_dim))
#       if L == 'Archery':
#         examples = g_model.predict(noise)
#       elif L == 'Basketball':
#         examples = g_model.predict(noise)
#       elif L == 'Biking':
#         examples = g_model.predict(noise)
#       elif L == 'CricketShot':
#         examples = g_model.predict(noise)
#       elif L == 'HorseRace':
#         examples = g_model.predict(noise)
#       elif L == 'IceDancing':
#         examples = g_model.predict(noise)
#       elif L == 'LongJump':
#         examples = g_model.predict(noise)
#       elif L == 'MilitaryParade':
#         examples = g_model.predict(noise)
#       elif L == 'PlayingTabla':
#         examples = g_model.predict(noise)
#       else:
#         print('Invalid frames')
  
#       save_examples(examples)

#       #Generate the video in avi
#       output_video_path = os.path.join(get_content_path(),'generated_videos', 'generated_video.avi')
#       if is_debug() == True:
#         print("output_video_path.......?" , output_video_path)

#       gen_video(output_video_path)

#       #Convert avi to mp4
#       clip = moviepy.VideoFileClip(output_video_path)
#       output_video_path_converted_mp4 = os.path.join(get_content_path(),'generated_videos', 'generated_video.mp4')
#       clip.write_videofile(output_video_path_converted_mp4)

#       #Show the video
#       video_file = open(output_video_path_converted_mp4, 'rb') #enter the filename with filepath
#       video_bytes = video_file.read() #reading the file
#       st.video(video_bytes, format='video/mp4', start_time=0) #displaying the video

#       for i in os.listdir(os.path.join(get_content_path(), "generated_images")):
#         st.write(os.path.join(get_content_path(), "generated_images/") + i)
#         st.image(cv2.imread(os.path.join(get_content_path(), "generated_images/") + i))   
#     else:
#        st.error("No Valid Labels identified. Rephrase the search") 
#        if is_debug() == True:
#           print("No Valid Labels identified. Rephrase the search") 

   
