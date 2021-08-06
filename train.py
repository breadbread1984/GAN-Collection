#!/usr/bin/python3

from os.path import join, exists;
from absl import app, flags;
import tensorflow as tf;
from models import *;
from datasets import *;

FLAGS = flags.FLAGS;
flags.DEFINE_enum('model', default = 'gan', enum_values = ['gan', 'dcgan'], help = 'models to train');
flags.DEFINE_integer('batch_size', default = 128, help = 'batch size');

def main(unused_argv):

  # 1) create dataset
  if FLAGS.model == 'gan':
    dataset = MNIST();
    trainset = dataset.load_trainset().map(gan.parse_function_generator()).batch(FLAGS.batch_size);
    testset = dataset.load_testset().map(gan.parse_function_generator()).batch(FLAGS.batch_size);
  elif FLAGS.model == 'dcgan':
    dataset = CelebA();
    trainset = dataset.load_dataset().map(dcgan.parse_function_generator(y_size = dataset.y_size)).batch(FLAGS.batch_size);
    testset = dataset.load_dataset().map(dcgan.parse_function_generator(y_size = dataset.y_size)).batch(FLAGS.batch_size);
  else:
    raise Exception('unknown model!');
  # 2) create or load compiled model
  if exists(join('checkpoints', FLAGS.model)):
    if FLAGS.model == 'gan':
      custom_objects = {'tf': tf, 'd_loss': gan.d_loss, 'g_loss': gan.g_loss};
    elif FLAGS.model == 'dcgan':
      custom_objects = {'tf': tf, 'd_loss': dcgan.d_loss, 'g_loss': dcgan.g_loss};
    else:
      raise Exception('unknown model!');
    model = tf.keras.models.load_model(join('checkpoints', FLAGS.model), custom_objects = custom_objects, compile = True);
    optimizer = model.optimizer;
  else:
    callbacks = [
      tf.keras.callbacks.TensorBoard(log_dir = join('checkpoints', FLAGS.model)),
      tf.keras.callbacks.ModelCheckpoint(filepath = join('checkpoints', FLAGS.model), save_freq = 1000)
    ];
    if FLAGS.model == 'gan':
      model = gan.Trainer();
      callbacks.append(gan.SummaryCallback(model));
      loss = {'d_loss': gan.d_loss, 'g_loss': gan.g_loss};
    elif FLAGS.model == 'dcgan':
      model = dcgan.Trainer(y_size = dataset.y_size);
      loss = {'d_loss': dcgan.d_loss, 'g_loss': dcgan.g_loss};
    else:
      raise Exception('unknown model!');
    optimizer = tf.keras.optimizers.Adam(1e-3);
    model.compile(optimizer = optimizer, loss = loss);
  # 3) train the model
  model.fit(trainset, epochs = 500, validation_data = testset, callbacks = callbacks);

if __name__ == "__main__":

  app.run(main);
