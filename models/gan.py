#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def Generator(z_size = 100, h1_size = 150, h2_size = 300, img_size = (28, 28)):

  assert type(img_size) in [list, tuple] and len(img_size) == 2;
  z_prior = tf.keras.Input((z_size,)); # z_prior.shape = (batch, 100)
  h1 = tf.keras.layers.Dense(h1_size, activation = tf.keras.activations.relu, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.1))(z_prior);
  h2 = tf.keras.layers.Dense(h2_size, activation = tf.keras.activations.relu, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.1))(h1);
  x_generate = tf.keras.layers.Dense(np.prod(img_size), activation = tf.keras.activations.tanh, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.1))(h2);
  x_generate = tf.keras.layers.Reshape(img_size)(x_generate); # x_generate.shape = (batch, 28, 28)
  return tf.keras.Model(inputs = z_prior, outputs = x_generate);

def Discriminator(z_size = 100, h1_size = 150, h2_size = 300, img_size = (28, 28), drop_rate = 0.2):
  
  x = tf.keras.Input(img_size); # x.shape = (batch, h, w)
  x_flatten = tf.keras.layers.Flatten()(x); # x_flatten.shape = (batch, 28 * 28)
  h1 = tf.keras.layers.Dense(h2_size, activation = tf.keras.activations.relu, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.1))(x_flatten);
  h1 = tf.keras.layers.Dropout(rate = drop_rate)(h1);
  h2 = tf.keras.layers.Dense(h1_size, activation = tf.keras.activations.relu, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.1))(h1);
  h2 = tf.keras.layers.Dropout(rate = drop_rate)(h2);
  h3 = tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.1))(h2); # h3.shape = (batch, 1)
  return tf.keras.Model(inputs = x, outputs = h3);

def Trainer(z_size = 100, h1_size = 150, h2_size = 300, img_size = (28, 28), drop_rate = 0.2):

  z_prior = tf.keras.Input((z_size,)); # z_prior.shape = (batch_z, 100)
  x_generate = Generator(z_size = z_size, h1_size = h1_size, h2_size = h2_size, img_size = img_size)(z_prior);
  x_nature = tf.keras.Input(img_size); # x_nature.shape = (batch_x, 28, 28)
  x = tf.keras.layers.Concatenate(axis = 0)([x_generate, x_nature]);
  pred = Discriminator(z_size = z_size, h1_size = h1_size, h2_size = h2_size, img_size = img_size, drop_rate = drop_rate)(x); # pred.shape = (batch_z + batch_x, 1)
  # pred_generate.shape = (batch_z, 1), pred_nature.shape = (batch_x, 1)
  pred_generate, pred_nature = tf.keras.layers.Lambda(lambda x: tf.split(x[0], [tf.shape(x[1])[0], tf.shape(x[2])[0]], axis = 0))([pred, z_prior, x_nature]);
  return tf.keras.Model(inputs = (z_prior, x_nature), outputs = (pred_generate, pred_nature));

if __name__ == "__main__":

  z_prior = np.random.normal(size = (4,100));
  generator = Generator();
  discriminator = Discriminator();
  generated = generator(z_prior);
  pred = discriminator(generated);
  generator.save('generator.h5');
  discriminator.save('discriminator.h5');
  images = np.random.normal(size = (6, 28, 28));
  trainer = Trainer();
  pred_gen, pred_na = trainer([z_prior, images]);
  print(pred_gen.shape, pred_na.shape);
