"""
ref: https://raw.githubusercontent.com/tensorflow/models/master/inception/inception/imagenet_data.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os

class Dataset(object):
  """abstract dataset class"""
  """A simple class for handling data sets."""
  __metaclass__ = ABCMeta

  def __init__(self, name, subset, data_dir):
    """Initialize dataset using a subset and the path to the data."""
    assert subset in self.available_subsets(), self.available_subsets()
    self.name = name
    self.subset = subset
    self.data_dir = data_dir

  @abstractmethod
  def num_classes(self):
    """Returns the number of classes in the data set."""
    pass

  @abstractmethod
  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    pass

  @abstractmethod
  def download_message(self):
    """Prints a download message for the Dataset."""
    pass

  def available_subsets(self):
    """Returns the list of available subsets."""
    return ['train', 'validation']

  def data_files(self):
    """Returns a python list of all (sharded) data subset files.

    Returns:
      python list of all (sharded) data set files.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    tf_record_pattern = os.path.join(self.data_dir, '%s-*' % self.subset)
    data_files = tf.gfile.Glob(tf_record_pattern)
    if not data_files:
      print('No files found for dataset %s/%s at %s' % (self.name,
                                                        self.subset,
                                                        self.data_dir))

      self.download_message()
      exit(-1)
    return data_files

  def reader(self):
    """Return a reader for a single entry from the data set.

    See io_ops.py for details of Reader class.

    Returns:
      Reader object that reads the data set.
    """
    return tf.TFRecordReader()

class ImagenetData(Dataset):
  """ImageNet data set."""

  def __init__(self, subset, data_dir):
    super(ImagenetData, self).__init__('ImageNet', subset, data_dir)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 1000

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data set."""
    # Bounding box data consists of 615299 bounding boxes for 544546 images.
    if self.subset == 'train':
      return 1281167
    if self.subset == 'validation':
      return 50000

  def download_message(self):
    """Instruction to download and extract the tarball from imagenet website."""

    print('Failed to find any ImageNet %s files'% self.subset)
    print('')
    print('If you have already downloaded and processed the data, then make '
          'sure to set --data_dir to point to the directory containing the '
          'location of the sharded TFRecords.\n')
    print('If you have not downloaded and prepared the ImageNet data in the '
          'TFRecord format, you will need to do this at least once. This '
          'process could take several hours depending on the speed of your '
          'computer and network connection\n')
    print('Please see README.md for instructions on how to build '
          'the ImageNet dataset using download_and_preprocess_imagenet.\n')
    print('Note that the raw data size is 300 GB and the processed data size '
          'is 150 GB. Please ensure you have at least 500GB disk space.')
