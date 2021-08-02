#!/usr/bin/python3

from os.path import join, exists;
from absl import app, flags;
import tensorflow as tf;
from models import *;
from datasets import *;

FLAGS = flags.FLAGS;
flags.DEFINE_enum('model', default = 'gan', enum_values = ['gan',], help = 'models to train');
flags.DEFINE_integer('batch_size', default = 128, help = 'batch size');

def main(unused_argv):

  # 1) create dataset
  if FLAGS.model == 'gan':
    trainset = dataset.load_trainset().map(gan.parse_function_generator()).batch(FLAGS.batch_size);
    testset = dataset.load_testset().map(gan.parse_function_generator()).batch(FLAGS.batch_size);
  else:
    raise Exception('unknown model!');
  # 2) create or load compiled model
  if exists(join('checkpoints', FLAGS.model)):
    if FLAGS.model == 'gan':
      custom_objects = {'tf': tf, 'd_loss': gan.d_loss, 'g_loss': gan.g_loss};
    else:
      raise Exception('unknown model!');
    model = tf.keras.models.load_model(join('checkpoints', FLAGS.model), custom_objects = custom_objects, compile = True);
    optimizer = model.optimizer;
  else:
    if FLAGS.model == 'gan':
      model = gan.Trainer();
      dataset = MNIST();
    else:
      raise Exception('unknown model!');
    optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.ExponentialDecay(1e-4, decay_steps = 20000, decay_rate = 0.97));
    model.compile(optimizer = optimizer,
                  loss = [gan.d_loss, gan.g_loss],
                  metrics = [tf.keras.metrics.Mean(), tf.keras.metrics.Mean()]);
  # 3) train the model
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = join('checkpoints', FLAGS.model)),
    tf.keras.callbacks.ModelCheckpoint(filepath = join('checkpoints', FLAGS.model), save_freq = 1000)
  ];
  model.fit(trainset, epochs = 500, validation_data = testset, callbacks = callbacks);

if __name__ == "__main__":

  app.run(main);
