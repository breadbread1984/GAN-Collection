#!/usr/bin/python3

from os import listdir;
from os.path import join;
from random import shuffle;
import pickle;
import cv2;
import tensorflow as tf;

class MNIST(object):
  def __init__(self,):
    (self.train_x, _), (self.test_x, _) = tf.keras.datasets.mnist.load_data();
  def load_trainset(self,):
    return tf.data.Dataset.from_tensor_slices(self.train_x);
  def load_testset(self,):
    return tf.data.Dataset.from_tensor_slices(self.test_x);

class CelebA(object):
  def __init__(self,):
    pass;
  def create_dataset(self, root_dir):
    name_id = dict();
    count = 0;
    samplelist = list();
    for people in listdir(root_dir):
      name_id[people] = count;
      for pic in listdir(join(root_dir, people)):
        samplelist.append((join(root_dir, people, pic), count));
      count += 1;
    with open('celeba_meta.pkl', 'wb') as f:
      f.write(pickle.dumps(name_id));
    shuffle(samplelist);
    writer = tf.io.TFRecordWriter('celeba_dataset.tfrecord');
    for img_path, cls in samplelist:
      img = cv2.imread(img_path);
      if img is None:
        print('can\'t process image %s' % img_path);
        continue;
      height, width, channel = img.shape;
      subimg = img[(height-128)//2:(height-128)//2+128,(width-128)//2:(width-128)//2+128,:];
      trainsample = tf.train.Example(features = tf.train.Features(
        feature = {
          'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.encode_jpeg(subimg).numpy()])),
          'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [cls])),
        }
      ));
      writer.write(trainsample.SerializeToString());
    self.y_size = count;
    return count;
  def load_dataset(self, filename = 'celeba_dataset.tfrecord', meta = 'celeba_meta.pkl'):
    with open(meta, 'rb') as f:
      name_id = pickle.loads(f.read());
    self.y_size = len(name_id.keys());
    return tf.data.TFRecordDataset(filename);

if __name__ == "__main__":
  from sys import argv;
  if len(argv) != 3:
    print('Usage: %s <dataset_name> <dataset_rootdir>' % argv[0]);
    exit(0);
  assert argv[1] in ['celeba',];
  if argv[1] == 'celeba':
    celeba = CelebA();
    celeba.create_dataset(argv[2]);
  else:
    raise Exception('unknown dataset!');
