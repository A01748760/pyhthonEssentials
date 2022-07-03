# Given an image, classify it using tags using tensorflow convolutional neuronal networks 
# Hint: run in jupyter notebook to see graphical content
# David Rodriguez

''' 
There are 10 neurones in the exit layer
There are 789 neurones in the first layer (28x28px)
'''

#import libraries
import math
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

#print("-------------------------------------------------------------------------")
#print(data)

training_data, tests_data = data['train'], data['test']


class_names = metadata.features['label'].names
#print(class_names)

#Normalize data from 0-255 to 0-1
def normalize(images, tags):
    images = tf.cast(images, tf.float32)
    images /= 255 #normalization operation
    return images, tags

# Normalize data using the normalize function we created
training_data = training_data.map(normalize)
tests_data = tests_data.map(normalize)

# Add to cache (to use memory instead of disk and achieve a faster training)
training_data = training_data.cache()
tests_data = tests_data.cache()

# show a sample image from the tests data
for image, tag in training_data.take(1):
    break
image = image.numpy().reshape((28,28)) #Resize 

# Draw sample
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

#draw example with tags
plt.figure(figsize=(10,10))
for i, (image, tag) in enumerate(training_data.take(25)):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[tag])
plt.show()

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)), #1 because its black/white
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) #softmax to ensure always an addition result of 1 

])

# compile
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), # function is commonly used in classificarion problems
    metrics=['accuracy']
)

ntraining = metadata.splits['train'].num_examples
ntests = metadata.splits['test'].num_examples

print(f'Ntraining: {ntraining}')
print(f'Ntests: {ntests}')


# optimization for faster execution
chunk_size = 32

training_data = training_data.repeat().shuffle(ntraining).batch(chunk_size)
tests_data = tests_data.batch(chunk_size)

#training (use fit)
record = model.fit(training_data, epochs=5, steps_per_epoch=math.ceil(ntraining/chunk_size))

# plot magnitude loss function
plt.xlabel("# Epoche")
plt.ylabel("Magnitude of loss")
plt.plot(record.history["loss"])