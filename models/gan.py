#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def Generator(z_size = 100, img_size = (28, 28)):

  assert type(img_size) in [list, tuple] and len(img_size) == 2;
  inputs = tf.keras.Input((z_size,)); # z_prior.shape = (batch, 100)
  results = tf.keras.layers.Dense(256)(inputs);
  results = tf.keras.layers.LeakyReLU(alpha = 0.2)(results);
  results = tf.keras.layers.BatchNormalization(momentum = 0.8)(results);
  results = tf.keras.layers.Dense(512)(results);
  results = tf.keras.layers.LeakyReLU(alpha = 0.2)(results);
  results = tf.keras.layers.BatchNormalization(momentum = 0.8)(results);
  results = tf.keras.layers.Dense(1024)(results);
  results = tf.keras.layers.LeakyReLU(alpha = 0.2)(results);
  results = tf.keras.layers.BatchNormalization(momentum = 0.8)(results);
  results = tf.keras.layers.Dense(np.prod(img_size), activation = tf.keras.activations.tanh)(results);
  results = tf.keras.layers.Reshape(img_size)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Discriminator(img_size = (28, 28)):
  
  inputs = tf.keras.Input(img_size); # x.shape = (batch, h, w)
  results = tf.keras.layers.Flatten()(inputs); # x_flatten.shape = (batch, 28 * 28)
  results = tf.keras.layers.Dense(512)(results);
  results = tf.keras.layers.LeakyReLU(alpha = 0.2)(results);
  results = tf.keras.layers.Dense(256)(results);
  results = tf.keras.layers.LeakyReLU(alpha = 0.2)(results);
  results = tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Trainer(z_size = 100, img_size = (28, 28)):

  z_prior = tf.keras.Input((z_size,)); # z_prior.shape = (batch_z, 100)
  x_generate = Generator(z_size = z_size, img_size = img_size)(z_prior);
  x_nature = tf.keras.Input(img_size); # x_nature.shape = (batch_x, 28, 28)
  # NOTE: stop_gradient is to prevent back propagation of d_loss from updating parameters of generator
  x = tf.keras.layers.Lambda(lambda x: tf.concat([tf.stop_gradient(x[0]), x[1]], axis = 0))([x_generate, x_nature]);
  disc = Discriminator(img_size = img_size);
  pred = disc(x); # pred.shape = (batch_z + batch_x, 1)
  # pred_generate.shape = (batch_z, 1), pred_nature.shape = (batch_x, 1)
  pred_generate, pred_nature = tf.keras.layers.Lambda(lambda x: tf.split(x[0], [tf.shape(x[1])[0], tf.shape(x[2])[0]], axis = 0))([pred, z_prior, x_nature]);
  d_loss_real = tf.keras.layers.Lambda(lambda x: tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(x),x))(pred_nature);
  d_loss_fake = tf.keras.layers.Lambda(lambda x: tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.zeros_like(x),x))(pred_generate);
  d_loss = tf.keras.layers.Lambda(lambda x: 0.5 * (x[0] + x[1]), name = 'd_loss')([d_loss_real, d_loss_fake]);
  # NOTE: using untrainable discriminator is to prevent back propagation of g_loss from updating parameters of discriminator
  const_dist = tf.keras.Model(inputs = disc.inputs, outputs = disc.outputs); const_dist.trainable = False;
  pred_generate = const_dist(x_generate);
  g_loss = tf.keras.layers.Lambda(lambda x: tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(x),x), name = 'g_loss')(pred_generate);
  return tf.keras.Model(inputs = (z_prior, x_nature), outputs = (d_loss, g_loss));

def parse_function_generator(z_size = 100):
  def parse_function(sample):
    sample = tf.cast(sample, dtype = tf.float32);
    sample = sample / 255. * 2 - 1; # sample range in [-1, 1]
    # z_prior.shape = (100,), sample.shape = (28, 28)
    return (tf.random.normal(shape = (z_size,)), sample), {'d_loss': 0, 'tf.cast_11': 0};
  return parse_function;

def d_loss(_, d_loss):
  return d_loss;

def g_loss(_, g_loss):
  return g_loss;

class SummaryCallback(tf.keras.callbacks.Callback):
  def __init__(self, trainer, z_size = 100, eval_freq = 100):
    self.trainer = trainer;
    self.z_size = z_size;
    self.eval_freq = eval_freq;
    self.log = tf.summary.create_file_writer('checkpoints/gan');
  def on_batch_end(self, batch, logs = None):
    if batch % self.eval_freq == 0:
      inputs = tf.keras.Input((self.z_size,));
      results = self.trainer.layers[1](inputs);
      generator = tf.keras.Model(inputs = inputs, outputs = results); # generator.shape = (1, 28, 28)
      sample = generator(tf.random.normal(shape = (1, self.z_size,)));
      image = tf.cast((sample + 1) / 2 * 255, dtype = tf.uint8);
      image = tf.tile(tf.expand_dims(image, axis = -1), (1,1,1,3));
      with self.log.as_default():
        tf.summary.image('generated', image, step = self.trainer.optimizer.iterations);

if __name__ == "__main__":

  z_prior = np.random.normal(size = (6,100));
  generator = Generator();
  discriminator = Discriminator();
  generated = generator(z_prior);
  pred = discriminator(generated);
  generator.save('generator.h5');
  discriminator.save('discriminator.h5');
  images = np.random.normal(size = (6, 28, 28));
  trainer = Trainer();
  d_loss, g_loss = trainer([z_prior, images]);
  trainer.save('trainer.h5');
  print(d_loss.shape, g_loss.shape);

