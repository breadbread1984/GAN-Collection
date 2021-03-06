#!/usr/bin/python3

from math import ceil;
import numpy as np;
import tensorflow as tf;

def Generator(z_size = 100, g_channel = 64, img_channel = 3, class_num = None, y_size = None, gc_channel = 1024, img_size = (64, 64)):
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
    h0 = tf.keras.layers.Dense(8 * g_channel * np.prod(shape16))(z); # y_h0.shape = (batch, (64 * 8) * 4 * 4)
    h0 = tf.keras.layers.Reshape((shape16[0], shape16[1], 8 * g_channel))(h0); # y_h0.shape = (batch, 4, 4, 64 * 8)
    h0 = tf.keras.layers.BatchNormalization()(h0);
    h0 = tf.keras.layers.ReLU()(h0);
    h1 = tf.keras.layers.Conv2DTranspose(4 * g_channel, kernel_size = (5,5), strides = (2,2), padding = 'same')(h0); # y_h0.shape = (battch, 8, 8, 64 * 4)
    h1 = tf.keras.layers.BatchNormalization()(h1);
    h1 = tf.keras.layers.ReLU()(h1);
    h2 = tf.keras.layers.Conv2DTranspose(2 * g_channel, kernel_size = (5,5), strides = (2,2), padding = 'same')(h1); # y_h1.shape = (batch, 16, 16, 64 * 2)
    h2 = tf.keras.layers.BatchNormalization()(h2);
    h2 = tf.keras.layers.ReLU()(h2);
    h3 = tf.keras.layers.Conv2DTranspose(1 * g_channel, kernel_size = (5,5), strides = (2,2), padding = 'same')(h2); # y_h3.shape = (batch, 32, 32, 64 * 1)
    h3 = tf.keras.layers.BatchNormalization()(h3);
    h3 = tf.keras.layers.ReLU()(h3);
    results = tf.keras.layers.Conv2DTranspose(img_channel, kernel_size = (5,5), strides = (2,2), padding = 'same', activation = tf.keras.activations.tanh)(h3); # y_h4.shape = (batch, 64, 64, 3)
  else:
    def calc_shapes(img_size):
      h, w = img_size[0], img_size[1];
      h2, w2 = ceil(h / 2), ceil(w / 2);
      h4, w4 = ceil(h2 / 2), ceil(w2 / 2);
      return (h2, w2), (h4, w4);
    # has condition input
    shape2, shape4 = calc_shapes(img_size);
    y = tf.keras.Input(()); # y.shape = (batch,)
    y_embed = tf.keras.layers.Embedding(class_num, y_size)(y);
    yb = tf.keras.layers.Reshape((1,1,y_size))(y_embed); # reshaped_y.shape = (batch, 1, 1, y_size)
    concated_z = tf.keras.layers.Concatenate(axis = -1)([z,y_embed]); # concated_z.shape = (batch, z_size + y_size)
    h0 = tf.keras.layers.Dense(gc_channel)(concated_z); # h0.shape = (batch, 1024)
    h0 = tf.keras.layers.BatchNormalization()(h0);
    h0 = tf.keras.layers.ReLU()(h0);
    h0 = tf.keras.layers.Concatenate(axis = -1)([h0, y_embed]); # h0.shape = (batch, 1024 + y_size)
    h1 = tf.keras.layers.Dense(2 * g_channel * np.prod(shape4))(h0); # h1.shape = (batch, 16 * 16 * 64 * 2)
    h1 = tf.keras.layers.BatchNormalization()(h1);
    h1 = tf.keras.layers.ReLU()(h1);
    h1 = tf.keras.layers.Reshape((shape4[0], shape4[1], 2 * g_channel))(h1); # h1.shape = (batch, 16, 16, 64 * 2)
    h1 = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], tf.tile(x[1], (1, tf.shape(x[0])[1], tf.shape(x[0])[2], 1))], axis = -1))([h1, yb]); # h1.shape = (batch, 16, 16, 64 * 2 + y_size)
    h2 = tf.keras.layers.Conv2DTranspose(2 * g_channel, kernel_size = (5,5), strides = (2,2), padding = 'same')(h1); # h2.shape = (batch, 32, 32, 64 * 2)
    h2 = tf.keras.layers.BatchNormalization()(h2);
    h2 = tf.keras.layers.ReLU()(h2);
    h2 = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], tf.tile(x[1], (1, tf.shape(x[0])[1], tf.shape(x[0])[2], 1))], axis = -1))([h2, yb]); # h2.shape = (batch, 32, 32, 64 * 2 + y_size)
    results = tf.keras.layers.Conv2DTranspose(img_channel, kernel_size = (5,5), strides = (2,2), padding = 'same', activation = tf.keras.activations.sigmoid)(h2); # h3.shape = (batch, 64, 64, 3)
  return tf.keras.Model(inputs = z if y_size is None else (z, y), outputs = results);

def Discriminator(d_channel = 64, img_channel = 3, class_num = None, y_size = None, dc_channel = 1024, img_size = (64, 64)):
  image = tf.keras.Input((img_size[0], img_size[1], img_channel)); # z.shape = (batch, h = 64, w = 64, 3)
  if y_size is None:
    h0 = tf.keras.layers.Conv2D(d_channel, kernel_size = (5,5), strides = (2,2), padding = 'same')(image); # h0.shape = (batch, 32, 32, 64)
    h0 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h0);
    h1 = tf.keras.layers.Conv2D(d_channel * 2, kernel_size = (5,5), strides = (2,2), padding = 'same')(h0); # h1.shape = (batch, 16, 16, 64)
    h1 = tf.keras.layers.BatchNormalization()(h1);
    h1 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h1);
    h2 = tf.keras.layers.Conv2D(d_channel * 4, kernel_size = (5,5), strides = (2,2), padding = 'same')(h1); # h2.shape = (batch, 8, 8, 64)
    h2 = tf.keras.layers.BatchNormalization()(h2);
    h2 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h2);
    h3 = tf.keras.layers.Conv2D(d_channel * 8, kernel_size = (5,5), strides = (2,2), padding = 'same')(h2); # h3.shape = (batch, 4, 4, 64)
    h3 = tf.keras.layers.BatchNormalization()(h3);
    h3 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h3);
    h3 = tf.keras.layers.Flatten()(h3); # h3.shape = (batch, 4 * 4 * 64)
    results = tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid)(h3); # results.shape = (batch, 1)
  else:
    y = tf.keras.Input(()); # y.shape = (batch,)
    y_embed = tf.keras.layers.Embedding(class_num, y_size)(y);
    yb = tf.keras.layers.Reshape((1,1,y_size))(y_embed); # reshaped_y.shape = (batch, 1, 1, y_size)
    x = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], tf.tile(x[1], (1, tf.shape(x[0])[1], tf.shape(x[0])[2], 1))], axis = -1))([image, yb]); # x.shape = (batch, 64, 64, 3 + y_size)
    h0 = tf.keras.layers.Conv2D(img_channel + y_size, kernel_size = (5,5), strides = (2,2), padding = 'same')(x); # h0.shape = (batch, 32, 32, 3 + y_size)
    h0 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h0);
    h0 = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], tf.tile(x[1], (1, tf.shape(x[0])[1], tf.shape(x[0])[2], 1))], axis = -1))([h0, yb]); # h0.shape = (batch, 32, 32, 3 + y_size + y_size)
    h1 = tf.keras.layers.Conv2D(d_channel + y_size, kernel_size = (5,5), strides = (2,2), padding = 'same')(h0); # h1.shape = (batch, 16, 16, 64 + y_size)
    h1 = tf.keras.layers.BatchNormalization()(h1);
    h1 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h1);
    h1 = tf.keras.layers.Flatten()(h1); # h1.shape = (batch, 16 * 16 * (64 + y_size))
    h1 = tf.keras.layers.Concatenate(axis = -1)([h1, y_embed]); # h1.shape = (batch, 16 * 16 * (64 + y_size) + y_size)
    h2 = tf.keras.layers.Dense(dc_channel)(h1); # h2.shape = (batch, 1024)
    h2 = tf.keras.layers.BatchNormalization()(h2);
    h2 = tf.keras.layers.LeakyReLU(alpha = 0.2)(h2);
    h2 = tf.keras.layers.Concatenate(axis = -1)([h2, y_embed]); # h2.shape = (batch, 1024 + y_size)
    results = tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid)(h2); # results.shape = (batch, 1)
  return tf.keras.Model(inputs = image if y_size is None else (image, y), outputs = results);

def Trainer(z_size = 100, g_channel = 64, d_channel = 64, img_channel = 3, class_num = None, y_size = None, gc_channel = 1024, dc_channel = 1024, img_size = (64, 64)):
  z_prior = tf.keras.Input((z_size,)); # z_prior.shape = (batch, 100)
  if y_size is not None: y_nature = tf.keras.Input(()); # y_nature.shape = (batch,)
  x_generate = Generator(z_size = z_size, g_channel = g_channel, img_channel = img_channel, class_num = class_num, y_size = y_size, gc_channel = gc_channel, img_size = img_size)(z_prior if y_size is None else [z_prior, y_nature]);
  x_nature = tf.keras.Input((img_size[0], img_size[1], img_channel)); # x_nature.shape = (batch, 28, 28, 3)
  # NOTE: stop_gradient is to prevent back propagation of d_loss from updating parameters of generator
  x = tf.keras.layers.Lambda(lambda x: tf.concat([tf.stop_gradient(x[0]), x[1]], axis = 0))([x_generate, x_nature]);
  cond = tf.keras.layers.Concatenate(axis = 0)([y_nature, y_nature]);
  disc = Discriminator(d_channel = d_channel, img_channel = img_channel, class_num = class_num, y_size = y_size, dc_channel = dc_channel, img_size = img_size);
  pred = disc(x if y_size is None else [x, cond]); # pred.shape = (batch_z + batch_x, 1)
  pred_generate, pred_nature = tf.keras.layers.Lambda(lambda x: tf.split(x[0], [tf.shape(x[1])[0], tf.shape(x[2])[0]], axis = 0))([pred, x_generate, x_nature]);
  d_loss_real = tf.keras.layers.Lambda(lambda x: tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(x),x))(pred_nature);
  d_loss_fake = tf.keras.layers.Lambda(lambda x: tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.zeros_like(x),x))(pred_generate);
  d_loss = tf.keras.layers.Lambda(lambda x: 0.5 * (x[0] + x[1]), name = 'd_loss')([d_loss_real, d_loss_fake]);
  # NOTE: enclosing discriminator within lambda function is to prevent back propagation of g_loss from updating parameters of discriminator
  const_dist = tf.keras.Model(inputs = disc.inputs, outputs = disc.outputs); const_dist.trainable = False;
  pred_generate = const_dist(x_generate if y_size is None else [x_generate, y_nature]);
  g_loss = tf.keras.layers.Lambda(lambda x: tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(x),x), name = 'g_loss')(pred_generate);
  return tf.keras.Model(inputs = (z_prior, x_nature) if y_size is None else (z_prior, x_nature, y_nature), outputs = (d_loss, g_loss));

def parse_function_generator(z_size = 100):
  def parse_function(serialized_example):
    feature = tf.io.parse_single_example(
      serialized_example,
      features = {
        'image': tf.io.FixedLenFeature((), dtype = tf.string),
        'label': tf.io.FixedLenFeature((), dtype = tf.int64)
      });
    sample = tf.io.decode_jpeg(feature['image']);
    sample = tf.image.resize(sample, (64, 64));
    label = feature['label'];
    sample = tf.cast(sample, dtype = tf.float32);
    sample = sample / 255. * 2 - 1; # sample range in [-1, 1]
    # z_prior.shape = (100,), sample.shape = (28, 28)
    return (tf.random.normal(shape = (z_size,)), sample, label), {'d_loss': 0, 'g_loss': 0};
  return parse_function;

def d_loss(_, d_loss):
  return d_loss;

def g_loss(_, g_loss):
  return g_loss;

class SummaryCallback(tf.keras.callbacks.Callback):
  def __init__(self, trainer, class_num = None, z_size = 100, eval_freq = 100):
    self.trainer = trainer;
    self.class_num = class_num;
    self.z_size = z_size;
    self.eval_freq = eval_freq;
    self.log = tf.summary.create_file_writer('checkpoints/dcgan');
  def on_batch_end(self, batch, logs = None):
    if batch % self.eval_freq == 0:
      z_prior = tf.keras.Input((self.z_size,));
      if self.class_num is not None: y_nature = tf.keras.Input(());
      results = self.trainer.layers[2]([z_prior, y_nature] if self.class_num is not None else z_prior);
      generator = tf.keras.Model(inputs = (z_prior, y_nature) if self.class_num is not None else z_prior, outputs = results);
      sample = generator([tf.random.normal(shape = (1, self.z_size,)), tf.random.uniform((1,), maxval = self.class_num, dtype = tf.int32)] if self.class_num is not None else tf.random.normal(shape = (1, self.z_size,)));
      image = tf.cast((sample + 1) / 2 * 255, dtype = tf.uint8);
      with self.log.as_default():
        tf.summary.image('generated', image, step = self.trainer.optimizer.iterations);

if __name__ == "__main__":
  z_prior = np.random.normal(size = (4,100));
  x_nature = np.random.normal(size = (4, 64, 64, 3));
  y_nature = np.random.normal(size = (4, 10));
  trainer = Trainer(class_num = 100,y_size = 10);
  d_loss, g_loss = trainer([z_prior, x_nature, y_nature]);
  print(d_loss.shape, g_loss.shape);
  trainer.save('trainer.h5');
