#!/usr/bin/python3

from math import ceil;
import numpy as np;
import tensorflow as tf;

def Generator(z_size = 100, g_channel = 64, img_channel = 3, y_size = None, gc_channel = 1024, img_size = (64, 64)):
  z = tf.keras.Input((z_size,));
  if y_size is None:
    def calc_shapes(img_size):
      assert type(img_size) in [list, tuple] and len(img_size) == 2;
      h, w = img_size[0], img_size[1];
      h2, w2 = ceil(h / 2), ceil(w / 2);
      h4, w4 = ceil(h2 / 2), ceil(w2 / 2);
      h8, w8 = ceil(h4 / 2), ceil(w4 / 2);
      h16, w16 = ceil(h8 / 2), ceil(h8 / 2);
      return (h2, w2), (h4, w4), (h8, w8), (h16, w16);    
    shape2, shape4, shape8, shape16 = calc_shapes(img_size);
    h0 = tf.keras.layers.Dense(8 * g_channel * np.prod(shape16), kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(z); # y_h0.shape = (batch, (64 * 8) * 4 * 4)
    h0 = tf.keras.layers.Reshape((shape16[0], shape16[1], 8 * g_channel))(h0); # y_h0.shape = (batch, 4, 4, 64 * 8)
    h0 = tf.keras.layers.BatchNormalization()(h0);
    h0 = tf.keras.layers.ReLU()(h0);
    h1 = tf.keras.layers.Conv2DTranspose(4 * g_channel, kernel_size = (5,5), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h0); # y_h0.shape = (battch, 8, 8, 64 * 4)
    h1 = tf.keras.layers.BatchNormalization()(h1);
    h1 = tf.keras.layers.ReLU()(h1);
    h2 = tf.keras.layers.Conv2DTranspose(2 * g_channel, kernel_size = (5,5), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h1); # y_h1.shape = (batch, 16, 16, 64 * 2)
    h2 = tf.keras.layers.BatchNormalization()(h2);
    h2 = tf.keras.layers.ReLU()(h2);
    h3 = tf.keras.layers.Conv2DTranspose(1 * g_channel, kernel_size = (5,5), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h2); # y_h3.shape = (batch, 32, 32, 64 * 1)
    h3 = tf.keras.layers.BatchNormalization()(h3);
    h3 = tf.keras.layers.ReLU()(h3);
    results = tf.keras.layers.Conv2DTranspose(img_channel, kernel_size = (5,5), strides = (2,2), padding = 'same', activation = tf.keras.activations.tanh, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h3); # y_h4.shape = (batch, 64, 64, 3)
  else:
    def calc_shapes(img_size):
      h, w = img_size[0], img_size[1];
      h2, w2 = ceil(h / 2), ceil(w / 2);
      h4, w4 = ceil(h2 / 2), ceil(w2 / 2);
      return (h2, w2), (h4, w4);
    # has condition input
    shape2, shape4 = calc_shapes(img_size);
    y = tf.keras.Input((y_size,)); # y.shape = (batch, y_size)
    yb = tf.keras.layers.Reshape((1,1,y_size))(y); # reshaped_y.shape = (batch, 1, 1, y_size)
    concated_z = tf.keras.layers.Concatenate(axis = -1)([z,y]); # concated_z.shape = (batch, z_size + y_size)
    h0 = tf.keras.layers.Dense(gc_channel, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(concated_z); # h0.shape = (batch, 1024)
    h0 = tf.keras.layers.BatchNormalization()(h0);
    h0 = tf.keras.layers.ReLU()(h0);
    h0 = tf.keras.layers.Concatenate(axis = -1)([h0, y]); # h0.shape = (batch, 1024 + y_size)
    h1 = tf.keras.layers.Dense(2 * g_channel * np.prod(shape4), kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h0); # h1.shape = (batch, 16 * 16 * 64 * 2)
    h1 = tf.keras.layers.BatchNormalization()(h1);
    h1 = tf.keras.layers.ReLU()(h1);
    h1 = tf.keras.layers.Reshape((shape4[0], shape4[1], 2 * g_channel))(h1); # h1.shape = (batch, 16, 16, 64 * 2)
    h1 = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], tf.tile(x[1], (1, tf.shape(x[0])[1], tf.shape(x[0])[2], 1))], axis = -1))([h1, yb]); # h1.shape = (batch, 16, 16, 64 * 2 + y_size)
    h2 = tf.keras.layers.Conv2DTranspose(2 * g_channel, kernel_size = (5,5), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h1); # h2.shape = (batch, 32, 32, 64 * 2)
    h2 = tf.keras.layers.BatchNormalization()(h2);
    h2 = tf.keras.layers.ReLU()(h2);
    h2 = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], tf.tile(x[1], (1, tf.shape(x[0])[1], tf.shape(x[0])[2], 1))], axis = -1))([h2, yb]); # h2.shape = (batch, 32, 32, 64 * 2 + y_size)
    results = tf.keras.layers.Conv2DTranspose(img_channel, kernel_size = (5,5), strides = (2,2), padding = 'same', activation = tf.keras.activations.sigmoid, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h2); # h3.shape = (batch, 64, 64, 3)
  return tf.keras.Model(inputs = z if y_size is None else (z, y), outputs = results);

def Discriminator(d_channel = 64, img_channel = 3, y_size = None, dc_channel = 1024, img_size = (64, 64)):
  image = tf.keras.Input((img_size[0], img_size[1], img_channel)); # z.shape = (batch, h = 64, w = 64, 3)
  if y_size is None:
    h0 = tf.keras.layers.Conv2D(d_channel, kernel_size = (5,5), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(image); # h0.shape = (batch, 32, 32, 64)
    h0 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h0);
    h1 = tf.keras.layers.Conv2D(d_channel * 2, kernel_size = (5,5), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h0); # h1.shape = (batch, 16, 16, 64)
    h1 = tf.keras.layers.BatchNormalization()(h1);
    h1 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h1);
    h2 = tf.keras.layers.Conv2D(d_channel * 4, kernel_size = (5,5), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h1); # h2.shape = (batch, 8, 8, 64)
    h2 = tf.keras.layers.BatchNormalization()(h2);
    h2 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h2);
    h3 = tf.keras.layers.Conv2D(d_channel * 8, kernel_size = (5,5), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h2); # h3.shape = (batch, 4, 4, 64)
    h3 = tf.keras.layers.BatchNormalization()(h3);
    h3 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h3);
    h3 = tf.keras.layers.Flatten()(h3); # h3.shape = (batch, 4 * 4 * 64)
    results = tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h3); # results.shape = (batch, 1)
  else:
    y = tf.keras.Input((y_size,)); # y.shape = (batch, y_size)
    yb = tf.keras.layers.Reshape((1,1,y_size))(y); # reshaped_y.shape = (batch, 1, 1, y_size)
    x = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], tf.tile(x[1], (1, tf.shape(x[0])[1], tf.shape(x[0])[2], 1))], axis = -1))([image, yb]); # x.shape = (batch, 64, 64, 3 + y_size)
    h0 = tf.keras.layers.Conv2D(img_channel + y_size, kernel_size = (5,5), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(x); # h0.shape = (batch, 32, 32, 3 + y_size)
    h0 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h0);
    h0 = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], tf.tile(x[1], (1, tf.shape(x[0])[1], tf.shape(x[0])[2], 1))], axis = -1))([h0, yb]); # h0.shape = (batch, 32, 32, 3 + y_size + y_size)
    h1 = tf.keras.layers.Conv2D(d_channel + y_size, kernel_size = (5,5), strides = (2,2), padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h0); # h1.shape = (batch, 16, 16, 64 + y_size)
    h1 = tf.keras.layers.BatchNormalization()(h1);
    h1 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h1);
    h1 = tf.keras.layers.Flatten()(h1); # h1.shape = (batch, 16 * 16 * (64 + y_size))
    h1 = tf.keras.layers.Concatenate(axis = -1)([h1, y]); # h1.shape = (batch, 16 * 16 * (64 + y_size) + y_size)
    h2 = tf.keras.layers.Dense(dc_channel, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h1); # h2.shape = (batch, 1024)
    h2 = tf.keras.layers.BatchNormalization()(h2);
    h2 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h2);
    h2 = tf.keras.layers.Concatenate(axis = -1)([h2, y]); # h2.shape = (batch, 1024 + y_size)
    results = tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid, kernel_initializer = tf.keras.initializers.RandomNormal(stddev = 0.02))(h2); # results.shape = (batch, 1)
  return tf.keras.Model(inputs = image if y_size is None else (image, y), outputs = results);

def Trainer(z_size = 100, g_channel = 64, d_channel = 64, img_channel = 3, y_size = None, gc_channel = 1024, dc_channel = 1024, img_size = (64, 64)):
  z_prior = tf.keras.Input((z_size,)); # z_prior.shape = (batch, 100)
  if y_size is not None: y_nature = tf.keras.Input((y_size,)); # y_nature.shape = (batch, y_size)
  x_generate = Generator(z_size = z_size, g_channel = g_channel, img_channel = img_channel, y_size = y_size, gc_channel = gc_channel, img_size = img_size)(z_prior if y_size is None else [z_prior, y_nature]);
  x_nature = tf.keras.Input((img_size[0], img_size[1], img_channel)); # x_nature.shape = (batch, 28, 28, 3)
  x = tf.keras.layers.Concatenate(axis = 0)([x_generate, x_nature]);
  cond = tf.keras.layers.Concatenate(axis = 0)([y_nature, y_nature]);
  pred = Discriminator(d_channel = d_channel, img_channel = img_channel, y_size = y_size, dc_channel = dc_channel, img_size = img_size)(x if y_size is None else [x, cond]); # pred.shape = (batch_z + batch_x, 1)
  pred_generate, pred_nature = tf.keras.layers.Lambda(lambda x: tf.split(x[0], [tf.shape(x[1])[0], tf.shape(x[2])[0]], axis = 0))([pred, z_prior, x_nature]);
  d_true_labels = tf.keras.layers.Lambda(lambda x: tf.ones_like(x))(pred_nature);
  d_false_labels = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x))(pred_generate);
  g_true_labels = d_true_labels;
  d_loss_real = tf.keras.losses.BinaryCrossentropy(from_logits = False)(d_true_labels, pred_nature);
  d_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits = False)(d_false_labels, pred_generate);
  d_loss = tf.keras.layers.Add(name = 'd_loss')([d_loss_real, d_loss_fake]);
  g_loss = tf.keras.losses.BinaryCrossentropy(from_logits = False, name = 'g_loss')(g_true_labels, pred_generate);
  return tf.keras.Model(inputs = (z_prior, x_nature) if y_size is None else (z_prior, x_nature, y_nature), outputs = (d_loss, g_loss));

def parse_function_generator(z_size = 100, y_size = None):
  def parse_function(sample, label):
    sample = tf.cast(sample, dtype = tf.float32);
    sample = sample / 255. * 2 - 1; # sample range in [-1, 1]
    y = tf.one_hot(label, y_size); # y.shape = (y_size,)
    # z_prior.shape = (100,), sample.shape = (28, 28)
    return (tf.random.normal(shape = (z_size,)), sample, y), {'d_loss': 0, 'g_loss': 0};
  return parse_function;

def d_loss(_, d_loss):
  return d_loss;

def g_loss(_, g_loss):
  return g_loss;

if __name__ == "__main__":
  z_prior = np.random.normal(size = (4,100));
  x_nature = np.random.normal(size = (4, 64, 64, 3));
  y_nature = np.random.normal(size = (4, 10));
  trainer = Trainer(y_size = 10);
  d_loss, g_loss = trainer([z_prior, x_nature, y_nature]);
  print(d_loss.shape, g_loss.shape);
  trainer.save('trainer.h5');
