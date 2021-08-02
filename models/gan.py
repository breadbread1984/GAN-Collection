#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def Generator(z_size = 100, h1_size = 150, h2_size = 300, img_size = (28, 28)):

  assert type(img_size) in [list, tuple] and len(img_size) == 2;
  z_prior = tf.keras.Input((z_size,)); # z_prior.shape = (batch, 100)
  h1 = tf.keras.layers.Dense(h1_size, activation = tf.keras.activations.relu, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.1))(z_prior);
  h2 = tf.keras.layers.Dense(h2_size, activation = tf.keras.activations.relu, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.1))(h1);
  x_generate = tf.keras.layers.Dense(np.prod(img_size), activation = tf.keras.activations.tanh, kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.1))(h2);
  x_generate = tf.keras.layers.Reshape(img_size)(x_generate);
  return tf.keras.Input(inputs = z_prior, outputs = x_generate);

def Discriminator(img_size = (28, 28)):

  
  x_generate = tf.keras.Input(img_size);
