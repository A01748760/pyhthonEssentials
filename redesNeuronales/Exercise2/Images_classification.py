# Given an image, classify it using tags using tensorflow convolutional neuronal networks 
# David Rodriguez

#import libraries
import tensorflow as tf
import tensorflow_datasets as tfds

data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

training_data, tests_data = data['train'], data['test']

class_names = metadata.features['label'].names
class_names