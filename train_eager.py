#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
from absl import app, flags;
import tensorflow as tf;
from models import *;
from datasets import *;

FLAGS = flags.FLAGS;
flags.DEFINE_enum('model', default = 'gan', enum_values = ['gan', 'dcgan'], help = 'models to train');
flags.DEFINE_integer('batch_size', default = 32, help = 'batch size');
flags.DEFINE_integer('disc_train_steps', default = 5, help = 'discriminator training steps');
flags.DEFINE_integer('gen_train_steps', default = 1, help = 'generator trainig steps');
flags.DEFINE_integer('checkpoint_steps', default = 1000, help = 'how many steps for each checkpoint');
flags.DEFINE_integer('eval_steps', default = 100, help = 'how many steps for each evaluation');
flags.DEFINE_bool('save_model', default = False, help = 'whether to save model');

def main(unused_argv):
  # 1) create dataset
  if FLAGS.model == 'gan':
    dataset = MNIST();
    trainset = iter(dataset.load_trainset().map(gan.parse_function_generator()).batch(FLAGS.batch_size).repeat(-1));
    testset = iter(dataset.load_testset().map(gan.parse_function_generator()).batch(FLAGS.batch_size).repeat(-1));
  elif FLAGS.model == 'dcgan':
    dataset = CelebA();
    trainset = iter(dataset.load_dataset().map(dcgan.parse_function_generator()).batch(FLAGS.batch_size).repeat(-1));
    testset = iter(dataset.load_dataset().map(dcgan.parse_function_generator()).batch(FLAGS.batch_size).repeat(-1));
  else:
    raise Exception('unknown model!');
  # 2) create model
  if FLAGS.model == 'gan':
    generator = gan.Generator();
    discriminator = gan.Discriminator();
    lr = 2e-4;
  elif FLAGS.model == 'dcgan':
    generator = dcgan.Generator(class_num = dataset.class_num, y_size = 10);
    discriminator = dcgan.Discriminator();
    lr = 2e-4;
  else:
    raise Exception('unknown model!');
  # 3) optimizer
  optimizer = tf.keras.optimizers.Adam(lr);
  # 4) restore from existing checkpoint
  if not exists('checkpoints'): mkdir('checkpoints');
  if not exists(join('checkpoints', FLAGS.model)): mkdir(join('checkpoints', FLAGS.model));
  checkpoint = tf.train.Checkpoint(generator = generator, discriminator = discriminator, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint(join('checkpoints', FLAGS.model)));
  if FLAGS.save_model == True:
    if not exists('trained'): mkdir('trained');
    if not exists(join('trained', FLAGS.model)): mkdir(join('trained', FLAGS.model));
    generator.save(join('trained', FLAGS.model, 'generator.h5'));
    discriminator.save(join('trained', FLAGS.model, 'discriminator.h5'));
    exit();
  # 5) log
  log = tf.summary.create_file_writer('checkpoints');
  gen_loss = tf.keras.metrics.Mean(name = 'gen_loss', dtype = tf.float32);
  disc_loss = tf.keras.metrics.Mean(name = 'disc_loss', dtype = tf.float32);
  while True:
    example = next(trainset);
    inputs, _ = example;
    if FLAGS.model == 'gan':
      noises, natures = inputs;
    elif FLAGS.model == 'dcgan':
      noises, natures, labels = inputs;
    else:
      raise Exception('unknown model');
    if optimizer.iterations % (FLAGS.disc_train_steps + FLAGS.gen_train_steps) < FLAGS.disc_train_steps:
      with tf.GradientTape() as tape:
        if FLAGS.model == 'gan':
          fakes = generator(noises);
          nature_preds = discriminator(natures);
          fake_preds = discriminator(fakes);
        elif FLAGS.model == 'dcgan':
          fakes = generator([noises, labels]);
          nature_preds = discriminator([natures, labels]);
          fake_preds = discriminator([fakes, labels]);
        else:
          raise Exception('unknown model');
        d_loss = 0.5 * (tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(nature_preds), nature_preds) + \
                        tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.zeros_like(fake_preds), fake_preds));
      grads = tape.gradient(d_loss, discriminator.trainable_variables);
      optimizer.apply_gradients(zip(grads, discriminator.trainable_variables));
      disc_loss.update_state(d_loss);
    else:
      with tf.GradientTape() as tape:
        if FLAGS.model == 'gan':
          fakes = generator(noises);
          fake_preds = discriminator(fakes);
        elif FLAGS.model == 'dcgan':
          fakes = generator([noises, labels]);
          fake_preds = discriminator(fakes);
        g_loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(fake_preds), fake_preds);
      grads = tape.gradient(g_loss, generator.trainable_variables);
      optimizer.apply_gradients(zip(grads, generator.trainable_variables));
      gen_loss.update_state(g_loss);
    if tf.equal(optimizer.iterations % FLAGS.checkpoint_steps, 0):
      checkpoint.save(join('checkpoints', FLAGS.model, 'ckpt'));
    if tf.equal(optimizer.iterations % FLAGS.eval_steps, 0):
      with log.as_default():
        image = tf.cast((fakes[0:1,...] + 1) / 2 * 255, dtype = tf.uint8);
        image = tf.tile(tf.expand_dims(image, axis = -1), (1,1,1,3));
        tf.summary.scalar('d_loss', disc_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('g_loss', gen_loss.result(), step = optimizer.iterations);
        tf.summary.image('sample', image, step = optimizer.iterations);
        print('#%d: d_loss = %f g_loss = %f' % (optimizer.iterations, disc_loss.result(), gen_loss.result()));
        disc_loss.reset_states();
        gen_loss.reset_states();

if __name__ == "__main__":
  app.run(main);
