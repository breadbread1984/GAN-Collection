#!/usr/bin/python3

from os.path import join;
from absl import flags, app;
import numpy as np;
import cv2;
import tensorflow as tf;

FLAGS = flags.FLAGS;
flags.DEFINE_enum('model', default = 'gan', enum_values = ['gan', 'dcgan'], help = 'models to test');
flags.DEFINE_integer('condition', default = 0, help = 'condition for generator');

def test(unused_argv):
  generator = tf.keras.models.load_model(join('trained', FLAGS.model, 'generator.h5'));
  while True:
    if FLAGS.model == 'gan':
      noise = np.random.normal(size = (1,100));
      sample = generator(noise);
    elif FLAGS.model == 'dcgan':
      noise = np.random.normal(size = (1,100));
      sample = generator([noise, tf.reshape(FLAGS.condition, (1,1))]);
    image = tf.cast((sample + 1) / 2 * 255, dtype = tf.uint8);
    image = tf.tile(tf.expand_dims(image, axis = -1), (1,1,1,3));
    image = image[0].numpy();
    cv2.imshow('sample', image);
    cv2.waitKey();

if __name__ == "__main__":
  app.run(test);
